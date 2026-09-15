"""
nn_dn_sx.py — 用可学习/可微的层替换 dn + sx, 并让 rp 有效
==========================================================

现状 (C#\\legacy\\sjy.cs)
------------------------
dn()  :  jj = (int)(360/d2)                     固定等分角
        vcsArray[i] = AngleAxis(i*d2, v3) @ cs  等角扇区
sx()  :  每扇区 prmax = max||perp(Ap, vcs)||、rp = max_sector prmax   硬 max
        点按 s[i]*r[k] 缩放后投影到圆上                                硬分配

实测 (points.json):
  rp = 51.154373,  而点云 |p| 中位 ≈ 0.5      ->  rp 大了约 100 倍
  经 MapDoubleConeToLeaf 后 |uv| 中位 9.04e-5, 跨 3.94 dex   ->  点集严重失真
  透镜场 g ∈ [112, 11697] ≫ 2                 ->  Phi(g)=0, V ≡ 0

三个要求
--------
  (a) 覆盖:  rp >= max_i ||perp(p_i, v3)||
  (b) 最小:  rp 尽量小 (锥最紧)
  (c) 不失真 + 合锥: 共形映射后点集不失真, 且贴合构造出的圆锥

  三者互相拉扯: (b) 要 rp = max rho, (c) 要 rp = 几何均值 rho。
  只有把 rho 的分布【变窄】才能同时满足 —— 这正是分组层 (dn/sx) 的职责,
  也是它必须可学习而不是固定划分的理由。

  另一条关键: 覆盖 + 最小 联合 => rp 收敛到 soft_max(rho),
  而 soft_max(rho) 依赖 v3  =>  最小化 rp 就等于"找一个让柱面半径最小的轴",
  这就是"优化 rp 会带动 v3 自动变化"的准确含义。

本文件
------
  Axis            可学习 v3 (so(3) 指数映射, 无 normalize 奇点)
  AxisFromTh2     还原现状: v3 = v3(rp) 经 th2 —— 只有 1 个自由度 (y 恒为 0)
  SoftSectors     K 个可学习扇区中心 + von Mises 软分配 (用 (cos,sin) 内积, 无分支割线)
  soft_max        可微上界 (>= max)
  soft_quantile   可微分位数 (二分 + 隐函数定理给梯度)
  三个模式对比: 现状 / v3 自由 / v3 由 th2 耦合

运行: python nn_dn_sx.py
"""
import io, json, math, os, sys

import numpy as np
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

POINTS = r"C:\Users\23128\My project (2)\Assets\Resources\points.json"


# ======================================================================
#  基础工具
# ======================================================================
def hat(w):
    O = torch.zeros(3, 3)
    O[0, 1] = -w[2]; O[0, 2] = w[1]
    O[1, 0] = w[2];  O[1, 2] = -w[0]
    O[2, 0] = -w[1]; O[2, 1] = w[0]
    return O


def exp_so3(w):
    """Rodrigues: 无奇点, 处处光滑"""
    th = torch.linalg.norm(w)
    I = torch.eye(3)
    if float(th) < 1e-9:
        return I + hat(w)
    K = hat(w / th)
    return I + torch.sin(th) * K + (1.0 - torch.cos(th)) * (K @ K)


def unit(v, eps=1e-12):
    return v / torch.linalg.norm(v).clamp(min=eps)


class Axis(nn.Module):
    """可学习主轴 v3 ∈ S^2。参数是 so(3) 旋转向量 w, v3 = exp(w) @ v3_0。
       这样 v3 恒为单位向量, 且 w=0 处光滑 (不像 normalize(w) 在 0 处有奇点)。"""
    def __init__(self, v3_init):
        super().__init__()
        self.register_buffer("v3_0", torch.as_tensor(v3_init).clone())
        self.w = nn.Parameter(torch.zeros(3))

    def forward(self):
        return exp_so3(self.w) @ self.v3_0


def ortho_frame(v3):
    """返回 (e1, e2) 使 (e1, e2, v3) 右手正交。
       注意: 参考向量的切换是硬的(不光滑), 只在 v3 几乎平行于参考时触发;
       实际用 PCA 轴初始化, 一般不会触发 —— 这是本原型的已知不光滑点。"""
    a = torch.tensor([0.0, 0.0, 1.0])
    if abs(float(v3 @ a)) > 0.9:
        a = torch.tensor([1.0, 0.0, 0.0])
    e1 = unit(torch.linalg.cross(a, v3))
    e2 = torch.linalg.cross(v3, e1)
    return e1, e2


def cylinder_radius(P, v3):
    """柱面半径 rho_i = ||perp(p_i, v3)||; 返回 (rho, perp)"""
    proj = (P @ v3).unsqueeze(1) * v3
    perp = P - proj
    return torch.linalg.norm(perp, dim=1), perp


def soft_max(x, beta):
    """(1/beta) logsumexp(beta x)  >= max(x), 可微; beta 越大越接近 max"""
    return torch.logsumexp(beta * x, 0) / beta


def soft_quantile(x, q, tau=0.05, iters=60):
    """可微分位数:
       F(r) = Σ sigmoid((r - x_i)/tau) = q*N  的解 r。
       用二分求 r (no_grad), 再用隐函数定理 dr/dx_i = sigma'_i / Σ sigma'_j
       以直通方式把梯度接上 (数值上等于 r, 梯度是对。</の)"""
    N = x.numel()
    with torch.no_grad():
        lo, hi = float(x.min()), float(x.max())
        tgt = q * N
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if float(torch.sigmoid((mid - x) / tau).sum()) < tgt:
                lo = mid
            else:
                hi = mid
        r = 0.5 * (lo + hi)
    s = torch.sigmoid((r - x) / tau)
    d = s * (1.0 - s)
    w = d / d.sum().clamp(min=1e-12)
    return r + (w * (x - x.detach())).sum()      # 值 = r, 梯度 = w


class SoftSectors(nn.Module):
    """K 个可学习扇区中心方位角 + von Mises 软分配。
       用 cos(θ_i - φ_k) = cosθ_i cosφ_k + sinθ_i sinφ_k, 避免 atan2 的分支割线。"""
    def __init__(self, K=8, kappa_init=8.0):
        super().__init__()
        self.phi = nn.Parameter(torch.linspace(0, 2 * math.pi, K + 1)[:-1].clone())
        self.log_kappa = nn.Parameter(torch.tensor(math.log(kappa_init)))

    def forward(self, cos_t, sin_t):
        ck, sk = torch.cos(self.phi), torch.sin(self.phi)
        C = cos_t[:, None] * ck[None, :] + sin_t[:, None] * sk[None, :]   # (N,K)
        kap = torch.exp(self.log_kappa).clamp(1e-2, 5e3)
        return torch.softmax(kap * C, dim=1)                              # 软分配


class AxisFromTh2(nn.Module):
    """还原现状的耦合: v3 = v3(rp), 经 sjy.th2 的两个交点之差。
       MapDoubleConeToLeaf 里:
           h = rp*cos(d2),  H = pi*rp*sin(d2)
           fzj = atan2(0.5h, H)          (theta2 找不到实根时的兜底分支)
           c1 - c2 = ( 2*atan(H/s), 0, 2H*cot(theta) )
           v3 = normalize(c1 - c2)
       因为 c1 与 c2 的 y 分量都等于同一个 Y, 相减后 y 恒为 0
       =>  v3 永远落在 xz 平面里, 只有 1 个自由度。"""
    def __init__(self, d2_rad):
        super().__init__()
        self.d2 = d2_rad
        self.log_rp = nn.Parameter(torch.tensor(math.log(1.0)))

    def forward(self):
        rp = torch.exp(self.log_rp)
        h = rp * math.cos(self.d2)
        H = math.pi * rp * math.sin(self.d2)
        theta = math.atan2(0.5 * h, H)
        s, c = math.sin(theta), math.cos(theta)
        v = torch.stack([torch.tensor(2.0 * math.atan(H / s)),
                         torch.tensor(0.0),
                         torch.tensor(2.0 * H * c / s)])
        return unit(v), rp


# ======================================================================
#  损失
# ======================================================================
def loss_fn(rho, A, rp, d2_rad, w, scale=None):
    """rp 由 soft_max 构造, 覆盖约束自动成立, 所以损失里没有 cov 项。
       w = (w_min, w_dis, w_cone)"""
    s = scale if scale is not None else rho.detach().mean().clamp(min=1e-9)
    L_min = (rp / s) ** 2
    spread = torch.std(torch.log(rho.clamp(min=1e-30) / rp))
    L_dis = spread / math.sin(d2_rad)
    num = (A * rho[:, None]).sum(0)
    den = A.sum(0).clamp(min=1e-9)
    rho_k = num / den
    L_cone = (A * (rho[:, None] - rho_k[None, :]) ** 2).sum() / A.sum().clamp(min=1e-9)
    L = w[0] * L_min + w[1] * L_dis + w[2] * L_cone
    return L, dict(mn=float(L_min), dis=float(L_dis), cone=float(L_cone),
                   rho_std=float(spread), rho_mean=float(rho.mean()),
                   rho_max=float(rho.max()), rho_k=rho_k.detach())


def scan_axis_floor(P, d2_rad, n=4000, beta=60.0):
    """在 S^2 上扫 v3, 求 std(log rho) 与 soft_max(rho) 的可达下界。
       这直接回答: '把 rho 分布变窄' 这件事, 靠选轴到底能不能做到?"""
    idx = torch.arange(n, dtype=torch.float64)
    z = 1.0 - 2.0 * (idx + 0.5) / n
    th = math.pi * (1.0 + 5.0 ** 0.5) * idx
    r = torch.sqrt((1.0 - z * z).clamp(min=0.0))
    V = torch.stack([r * torch.cos(th), r * torch.sin(th), z], dim=1)
    best_std, best_lse = float("inf"), float("inf")
    worst_std = 0.0
    for v in V:
        rho, _ = cylinder_radius(P, v)
        sd = float(torch.std(torch.log(rho.clamp(min=1e-30))))
        ls = float(soft_max(rho, beta))
        best_std = min(best_std, sd)
        worst_std = max(worst_std, sd)
        best_lse = min(best_lse, ls)
    return best_std, worst_std, best_lse


# ======================================================================
#  主程序
# ======================================================================
def load_points(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        raw = list(raw.values())
    return np.asarray(raw, dtype=np.float64)


def pca_axis(P, already_centered=False):
    X = P - P.mean(0) if not already_centered else P
    ev, evec = np.linalg.eigh(np.cov(X.T))
    o = np.argsort(ev)[::-1]
    return evec[:, o[0]], ev[o] / ev[o].sum()


def report(tag, rho_np, rp_np, d2_rad, A_np, v3_np, rp_csharp=None):
    s = np.sin(d2_rad)
    # 与 C# MapDoubleConeToLeaf 的失真链条一致: Rs = (rho/rp)^(1/sin d2)
    Rs = (rho_np / rp_np) ** (1.0 / s)
    spread = (1.0 / s) * np.std(np.log(rho_np / rp_np))
    A = A_np
    rho_k = (A * rho_np[:, None]).sum(0) / np.maximum(A.sum(0), 1e-9)
    cone = float((A * (rho_np[:, None] - rho_k[None, :]) ** 2).sum() / max(A.sum(), 1e-9))
    print(f"\n  [{tag}]")
    print(f"    v3          = [{v3_np[0]:+.5f}, {v3_np[1]:+.5f}, {v3_np[2]:+.5f}]"
          f"   |v3|={np.linalg.norm(v3_np):.6f}")
    print(f"    rho         : min {rho_np.min():.5f}  med {np.median(rho_np):.5f}  "
          f"max {rho_np.max():.5f}   (max/med = {rho_np.max()/max(np.median(rho_np),1e-30):.2f})")
    print(f"    rp          = {rp_np:.6f}    rp/rho_max = {rp_np/max(rho_np.max(),1e-30):.6f}")
    print(f"    Rs=(rho/rp)^(1/sin d2) : min {Rs.min():.3e}  med {np.median(Rs):.3e}  "
          f"max {Rs.max():.3e}   跨度 {np.log10(Rs.max()/max(Rs.min(),1e-300)):.2f} dex")
    print(f"    失真度量 (1/sin d2)*std(log(rho/rp)) = {spread:.6f}")
    print(f"    合锥残差  Σ a(rho-rho_k)^2 / Σa      = {cone:.6f}")
    if rp_csharp is not None:
        print(f"    对照 C# 实测 rp = {rp_csharp:.6f}  ->  rp/rho_max = "
              f"{rp_csharp/max(rho_np.max(),1e-30):.2f}")


def main():
    P_np = load_points(POINTS)
    P = torch.as_tensor(P_np)
    axis_pca, vr = pca_axis(P_np)
    d2 = math.acos(float(np.clip(axis_pca[1], -1, 1)))
    print(f"points.json  N={len(P_np)}   |p| 中位 {np.median(np.linalg.norm(P_np,axis=1)):.5f}")
    print(f"PCA 轴 = [{axis_pca[0]:+.5f},{axis_pca[1]:+.5f},{axis_pca[2]:+.5f}]"
          f"   d2 = {math.degrees(d2):.4f} deg   方差比 = {vr}")
    print(f"1/sin(d2) = {1/math.sin(d2):.6f}   <- 共形映射把 log 跨度放大这么多倍")

    v3_0 = torch.as_tensor(axis_pca)
    K = 8
    BETA = 60.0
    weights = (1.0, 1.0, 1.0)          # min, dis, cone

    # ---------------- 可达下界: 换任何轴能做到多好 ----------------
    print("\n" + "=" * 78)
    print(f"可达下界扫描: 在 S^2 上取 4000 个方向当 v3")
    print("=" * 78)
    b_std, w_std, b_lse = scan_axis_floor(P, d2, n=4000, beta=BETA)
    print(f"  std(log rho) : 最好 {b_std:.6f}   最差 {w_std:.6f}   PCA 轴 "
          f"{float(torch.std(torch.log(cylinder_radius(P, unit(v3_0))[0]))):.6f}")
    print(f"  soft_max(rho): 最好 {b_lse:.6f}")
    print(f"  => 若【最好】与【最差】差别很小, 说明 rho 的分布宽度是内禀的,")
    print(f"     换任何轴都压不窄 —— 那分组层 (调 v3) 就救不了它。")

    # ---------------- 模式 1: 现状复现 (v3 = PCA 固定, 等角扇区, 硬 max) ----------------
    print("\n" + "=" * 78)
    print("模式 1  现状: v3 = PCA 固定, 等角扇区, rp = 硬 max")
    print("=" * 78)
    rho1, _ = cylinder_radius(P, unit(v3_0))
    rho1_np = rho1.detach().numpy()
    N = len(P)
    idx = torch.arange(N)
    th = (idx / N) * 2 * math.pi                       # dn 的等角扇区
    A1 = torch.zeros(N, K)
    A1[idx, (th / (2 * math.pi) * K).long() % K] = 1.0  # 硬分配
    report("现状", rho1_np, float(rho1.max()), d2, A1.numpy(), v3_0.numpy(), rp_csharp=51.154373)

    # ---------------- 模式 2: v3 自由 + 软扇区; rp = soft_max(rho) 构造 ----------------
    print("\n" + "=" * 78)
    print("模式 2  可学习: v3 ∈ S^2 自由, K 个软扇区, rp = soft_max(rho, beta) 构造")
    print("          => 覆盖 rp >= max rho 自动成立, 最小化 rp 等价于选轴让柱面半径最小")
    print("=" * 78)
    axis = Axis(axis_pca)
    sectors = SoftSectors(K)
    params = list(axis.parameters()) + list(sectors.parameters())
    opt = torch.optim.Adam(params, lr=5e-3)
    rho_ref = float(rho1.detach().mean())
    for it in range(4000):
        opt.zero_grad()
        v3 = axis()
        rho, _ = cylinder_radius(P, v3)
        e1, e2 = ortho_frame(v3)
        perp = P - (P @ v3).unsqueeze(1) * v3
        cos_t = (perp @ e1) / rho.clamp(min=1e-12)
        sin_t = (perp @ e2) / rho.clamp(min=1e-12)
        A = sectors(cos_t, sin_t)
        rp = soft_max(rho, BETA)
        L, st = loss_fn(rho, A, rp, d2, weights, scale=rho_ref)
        L.backward()
        opt.step()
        if it % 1000 == 0:
            print(f"    it {it:>4}  L={float(L):.6e}  rp={float(rp):.6f}  "
                  f"rho_max={st['rho_max']:.6f}  dis={st['dis']:.6f}  cone={st['cone']:.6f}")
    with torch.no_grad():
        v3b = axis(); rhob, _ = cylinder_radius(P, v3b)
        e1, e2 = ortho_frame(v3b); perpb = P - (P @ v3b).unsqueeze(1) * v3b
        cos_t = (perpb @ e1) / rhob.clamp(min=1e-12); sin_t = (perpb @ e2) / rhob.clamp(min=1e-12)
        Ab = sectors(cos_t, sin_t)
        rp_learn = float(soft_max(rhob, BETA))
    report("可学习", rhob.numpy(), rp_learn, d2, Ab.numpy(), v3b.numpy())

    # ---------------- 模式 3: v3 由 th2 与 rp 耦合 (现状的 1 自由度结构) ----------------
    print("\n" + "=" * 78)
    print("模式 3  现状结构: v3 = v3(rp) 经 th2  ——  y 恒为 0, 只有 1 个自由度")
    print("=" * 78)
    print("    验证 y 分量是否恒为 0:")
    for mult in [0.3, 1.0, 3.0, 10.0]:
        m = AxisFromTh2(d2)
        with torch.no_grad():
            m.log_rp.fill_(math.log(mult * rho_ref))
        v, r = m()
        print(f"      rp = {float(r):>9.4f}  ->  v3 = "
              f"[{float(v[0]):+.6f}, {float(v[1]):+.6e}, {float(v[2]):+.6f}]   y={float(v[1]):.1e}")
    print("    => v3.y 恒为 0, 轴被强制压在 xz 平面里, 自由度 = 1")
    print("       所以'可学习 v3'这件事, 必须先拆掉 MapDoubleConeToLeaf 里 v3 = normalize(c1-c2) 这层耦合")

    print("\n" + "=" * 78)
    print("小结")
    print("=" * 78)
    print(f"  1/sin(d2) = {1/math.sin(d2):.4f}: 共形映射 Rs=(rho/rp)^(1/sin d2) 把半径的 log 分布放大这么多倍。")
    print("      => rp 必须落在 rho 的分布中心附近, 否则 Rs 整体偏到 1e-4 或 1e4 量级, 点集被压扁。")
    print("  'rp 最小' 与 '不失真' 是冲突的: 前者要 rp = max rho, 后者要 rp = 几何均值 rho。")
    print("  唯一的出路是把 rho 的分布变窄 —— 这正是分组层 (替代 dn/sx) 的职责,")
    print("  也是它必须可学习的原因。")


if __name__ == "__main__":
    main()
