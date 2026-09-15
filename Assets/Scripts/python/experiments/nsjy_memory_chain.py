"""
nsjy_memory_chain.py — 记忆链: 数据 -> 复数 -> 角度 -> 概率 -> (t1,t2)
                            -> 模变换矩阵 -> 记忆矩阵 -> 位姿矩阵
================================================================================

按用户给出的架构落地。全链每一步都打印, 便于逐步核对。

链:
    (1) 数据 (n,4) ──加一个虚数 i ──▶ 5 个复数 = 10 维实数
    (2) 5 个复数 ──▶ 6 个角度
    (3) 三个锚角 d2 = 30 / 45 / 16  ──▶  τ = i*K(1-k)/K(k),  k = sin^2(d2)
        45° 给出 k = 1/2 => tau = i   (lemniscatic, 方形格)  <- 前面已验证
    (4) 每个复通道对 t30 与 t45 【各自】求格距离 -> 逐通道概率 -> 总概率
        (不再分别算 t30 和 t45 两套, 而是每个复数都对两个 t 求距离后合并)
    (5) 高斯格基约化(2D) 加速最近格点距离
    (6) 方向对齐 ──▶ 每个点的 (t1, t2)
    (7) 对 t30 / t45 求【模变换矩阵】(t1,t2) -> (t30,t45)
    (8) 模变换矩阵 + 李群 ──▶ 记忆矩阵, 可以从后往前逐渐铺开
    (9) 角度 + 记忆矩阵 ──▶ 位姿矩阵

标注 [假设] 的地方是我必须猜的, 需要用户确认。

运行: python nsjy_memory_chain.py
"""
import io, json, math, sys

import numpy as np
from scipy.special import ellipk

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PTS = r"C:\Users\23128\My project (2)\Assets\Resources\points.json"
ANCHOR_DEG = (30.0, 45.0, 16.0)      # 用户指定: 前三个角
SIGMA = 0.25                          # 格距离 -> 概率的尺度
NCH = 5                               # 5 个复通道 = 10 维


# =====================================================================
# (3) 锚角 -> tau
# =====================================================================
def tau_from_d2(d2_deg):
    """k = sin^2(d2);  tau = i*K(1-k)/K(k)。 (k 为模量)"""
    m = math.sin(math.radians(d2_deg)) ** 2     # 参数 m = sin^2(d2)
    Kk = float(ellipk(m))                        # scipy ellipk 取参数 m
    Kp = float(ellipk(1.0 - m))
    return 1j * (Kp / Kk), m


# =====================================================================
# (1) 嵌入: (n,4) -> (n,5) 复数   [假设: 第 5 个通道 = 恒等虚数 i 通道]
# =====================================================================
def embed(pts):
    n = pts.shape[0]
    z = np.zeros((n, NCH), dtype=complex)
    k = min(pts.shape[1], NCH - 1)
    for j in range(k):
        z[:, j] = pts[:, j] + 1j * pts[:, (j + 1) % pts.shape[1]]
    z[:, NCH - 1] = 1j * np.ones(n)     # "加一个虚数 1"
    return z


# =====================================================================
# (2) 5 个复数 -> 6 个角度    [假设: 5 个相位 + 1 个整体比例角]
# =====================================================================
def six_angles(z):
    n = z.shape[0]
    A = np.zeros((n, 6))
    A[:, :NCH] = np.angle(z)                                    # 5 个相位
    nz = np.abs(z)
    tot = nz.sum(axis=1)
    tot[tot < 1e-300] = 1e-300
    A[:, 5] = 2.0 * math.pi * nz[:, 0] / tot                     # 第 6 个: 主通道占比
    return A


# =====================================================================
# (5) 高斯格基约化 (2D) —— 用来加速最近格点距离
# =====================================================================
def gauss_reduce(b1, b2, max_iter=200):
    """2D 格基约化: 返回 |b1| <= |b2| 且 |Re(b2/b1)| <= 1/2 的基。"""
    if abs(b1) > abs(b2):
        b1, b2 = b2, b1
    for _ in range(max_iter):
        if abs(b1) < 1e-300:
            break
        mu = round((b2 * b1.conjugate()).real / (b1 * b1.conjugate()).real)
        if mu == 0:
            break
        b2 = b2 - mu * b1
        if abs(b1) > abs(b2):
            b1, b2 = b2, b1
    return b1, b2


def nearest_dist(z, tau):
    """到格 Z + tau Z 的最近距离 (欧氏), 用约化基加速。"""
    b1, b2 = gauss_reduce(1.0 + 0j, tau)
    M = np.array([[b1.real, b2.real], [b1.imag, b2.imag]])
    if abs(np.linalg.det(M)) < 1e-14:
        return float("inf")
    Minv = np.linalg.inv(M)
    ab = Minv @ np.array([z.real, z.imag])
    cands = set()
    for c0 in (math.floor(ab[0]), math.ceil(ab[0])):
        for c1 in (math.floor(ab[1]), math.ceil(ab[1])):
            for da in (-1, 0, 1):
                for db in (-1, 0, 1):
                    cands.add((c0 + da, c1 + db))
    return min(abs(a * b1 + b * b2 - z) for a, b in cands)


# =====================================================================
# (4) 逐通道概率 -> 总概率
# =====================================================================
def channel_prob(z, tau, sigma=SIGMA):
    return np.array([math.exp(-nearest_dist(zz, tau) ** 2 / (2.0 * sigma * sigma))
                     for zz in z])


def total_prob(z, taus, sigma=SIGMA):
    """对每个复通道【各自】到每个 tau 求距离 -> 概率 -> 合并。"""
    per = np.stack([channel_prob(z[:, j], t) for t in taus for j in range(NCH)], axis=0)
    return per.prod(axis=0) ** (1.0 / per.shape[0]), per


# =====================================================================
# (6) 方向对齐 -> 每个点的 (t1, t2)
# =====================================================================
def align_taus(z, taus, iters=60):
    """对每个点, 在锚点 tau 集合里找使概率最大的那个, 再在局部细化 tau。

    [假设] '方向对齐' 我落地为: 点在哪个锚 tile 的概率最大, 就把该 tile 的 tau
           作为该点的 (t1, t2) 起点, 再用相位微调。
    """
    n = z.shape[0]
    probs = np.stack([channel_prob(z[:, 0], t) for t in taus], axis=1)
    pick = np.argmax(probs, axis=1)
    t1 = np.array([taus[p] for p in pick], dtype=complex)
    t2 = np.array([taus[(p + 1) % len(taus)] for p in pick], dtype=complex)
    # 相位微调: 用第一通道相位把 tau 旋转一个小量 [假设]
    adj = 0.1 * np.angle(z[:, 0])
    t1 = t1 * np.exp(1j * adj)
    t2 = t2 * np.exp(-1j * adj)
    return t1, t2, pick


# =====================================================================
# (7) 模变换矩阵  (t1,t2) -> (t30,t45)
# =====================================================================
def modulus_matrix(t1, t2, t_anchor=(0, 1)):
    """对每个点求把 (t1,t2) 送到锚 (t30,t45) 的 SL(2,C) 矩阵。

    [假设] 用两个复数的比值构造规范形矩阵 [[t2, t1],[t2_anchor, t1_anchor]]
           的归一化形式; 这是模变换的标准写法 M = [[a,b],[c,d]], tau = (a*tau+b)/(c*tau+d)。
    """
    n = len(t1)
    out = np.zeros((n, 2, 2), dtype=complex)
    ta, tb = t_anchor
    for i in range(n):
        M = np.array([[tb, ta], [t2[i], t1[i]]], dtype=complex)
        # 极分解归一 -> 酉矩阵, 保证李群连乘有界 (否则爆到 1e48)
        G = M.conj().T @ M
        w, V = np.linalg.eigh(G)
        w = np.maximum(w, 1e-300)
        Ginv = V @ np.diag(1.0 / np.sqrt(w)) @ V.conj().T
        out[i] = M @ Ginv
    return out


# =====================================================================
# (8) 记忆矩阵: 李群元, 从后往前逐渐铺开
# =====================================================================
def memory_matrix(mods):
    """把逐点的模变换矩阵沿序列【从后往前】累积成记忆矩阵。

    [假设] 李群乘法顺序按用户说的 '从后往前逐渐铺开' 取右乘累积。
    """
    n = len(mods)
    acc = np.eye(2, dtype=complex)
    hist = np.zeros((n, 2, 2), dtype=complex)
    for i in range(n - 1, -1, -1):
        acc = acc @ mods[i]
        hist[i] = acc
    return acc, hist


# =====================================================================
# (9) 位姿矩阵 = 角度 ⊗ 记忆矩阵
# =====================================================================
def pose_matrix(angles, mem):
    """把 6 个角度做成 SO(3)/SE(3) 的旋转部分, 与记忆矩阵做张量积。

    [假设] 角度 -> 三个欧拉旋转(Rz Ry Rx), 记忆矩阵取实部 2x2 嵌入 SE(3)。
    """
    n = angles.shape[0]
    out = np.zeros((n, 6, 6))
    # Re(U) 对酉矩阵【不保正交】, 必须取它的最近正交因子(极分解), 否则位姿矩阵不闭合
    Rm = np.zeros((mem.shape[0], 2, 2))
    for i in range(mem.shape[0]):
        U_, s_, Vt_ = np.linalg.svd(np.real(mem[i]))
        Rm[i] = U_ @ Vt_
    for i in range(n):
        ax, ay, az = angles[i, 0], angles[i, 1], angles[i, 2]
        cx, sx = math.cos(ax), math.sin(ax)
        cy, sy = math.cos(ay), math.sin(ay)
        cz, sz = math.cos(az), math.sin(az)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx
        out[i, :3, :3] = R
        out[i, 3:6, 3:6] = R
        out[i, :2, :2] = out[i, :2, :2] @ Rm[i]       # 记忆矩阵介入
        out[i, 3:5, 3:5] = out[i, 3:5, 3:5] @ Rm[i]
    return out


def main():
    pts = np.array(json.load(io.open(PTS, encoding="utf-8")), dtype=float)
    print("=" * 80)
    print("0. 输入")
    print("=" * 80)
    print(f"  points.json: {pts.shape[0]} 点 x {pts.shape[1]} 列")
    print()

    print("=" * 80)
    print("(3) 三个锚角 -> tau   (k = sin^2(d2),  tau = i*K(1-k)/K(k), 椭参数约定)")
    print("=" * 80)
    taus = []
    print(f"  {'d2':>7} {'k = sin^2(d2)':>14} {'K(m=k^2)':>12} {'K(1-m)':>12} "
          f"{'tau':>22} {'K(1-m)/K(m)':>12}")
    for d in ANCHOR_DEG:
        t, k = tau_from_d2(d)
        Km = float(ellipk(k))
        Kp = float(ellipk(1.0 - k))
        taus.append(t)
        print(f"  {d:>7.2f} {k:>14.6f} {Km:>12.6f} {Kp:>12.6f} "
              f"{t.real:>10.6f}{t.imag:>+11.6f}i {Kp/Km:>12.6f}")
    print()
    print("  注: 45° -> k = 0.5 -> K(1-m)/K(m) = 1.000000 -> tau = i  (方形格, lemniscatic)")
    print("      这正是前面 verify_three_foci.py / verify_two_sheet_closed_form.py 里的临界点。")
    print()

    print("=" * 80)
    print("(1) 嵌入: 加一个虚数 i -> 5 个复数 = 10 维")
    print("=" * 80)
    z = embed(pts)
    print(f"  形状: {z.shape}   (n x 5 复 = n x 10 实)")
    print(f"  第 1 点 5 个复数: " + ", ".join(f"{v.real:+.4f}{v.imag:+.4f}i" for v in z[0]))
    print()

    print("=" * 80)
    print("(2) 5 个复数 -> 6 个角度 (前 3 个)")
    print("=" * 80)
    A = six_angles(z)
    print(f"  {'点':>4} " + " ".join(f"{'a'+str(i):>11}" for i in range(6)))
    for i in range(min(4, len(A))):
        print(f"  {i:>4} " + " ".join(f"{A[i,j]:>11.5f}" for j in range(6)))
    print()

    print("=" * 80)
    print("(5) 高斯格基约化: 最近格点距离加速")
    print("=" * 80)
    for t in taus:
        b1, b2 = gauss_reduce(1.0 + 0j, t)
        # 暴力对照
        zz = z[0, 0]
        brute = min(abs(zz - (a + b * t)) for a in range(-30, 31) for b in range(-30, 31))
        fast = nearest_dist(zz, t)
        print(f"  tau = {t.real:.4f}{t.imag:+.4f}i  约化基 |b1|={abs(b1):.4f} "
              f"|b2|={abs(b2):.4f}  暴力={brute:.9f}  约化={fast:.9f}  差={abs(brute-fast):.2e}")
    print()

    print("=" * 80)
    print("(4) 逐通道 x 逐 tau 求距离 -> 总概率  (不再分两套)")
    print("=" * 80)
    print("  对【每个复通道】都求到 t30 和 t45 的距离, 再合并成总概率")
    for name, tt in (("t30", taus[0]), ("t45", taus[1])):
        p = channel_prob(z[:, 0], tt)
        print(f"  {name}: 通道0 概率 中位 {np.median(p):.6f}  "
              f"均值 {p.mean():.6f}  非零比例 {(p>1e-6).mean():.3f}")
    tot, per = total_prob(z, taus[:2])
    print(f"  总概率: 形状 {tot.shape}  中位 {np.median(tot):.6e}  "
          f"最大 {tot.max():.6e}  参与格点对数 {per.shape[0]}")
    print()

    print("=" * 80)
    print("(6) 方向对齐 -> 每点 (t1, t2)")
    print("=" * 80)
    t1, t2, pick = align_taus(z, taus)
    print(f"  锚点选择分布: " + ", ".join(
        f"{ANCHOR_DEG[j]}°x{(pick==j).sum()}" for j in range(len(ANCHOR_DEG))))
    print(f"  {'点':>4} {'t1':>22} {'t2':>22}")
    for i in range(min(4, len(t1))):
        print(f"  {i:>4} {t1[i].real:>10.6f}{t1[i].imag:>+11.6f}i "
              f"{t2[i].real:>10.6f}{t2[i].imag:>+11.6f}i")
    print()

    print("=" * 80)
    print("(7) 模变换矩阵 (t1,t2) -> (t30,t45)")
    print("=" * 80)
    mods = modulus_matrix(t1, t2, t_anchor=(taus[0], taus[1]))
    print(f"  M[0] = " + str(np.round(mods[0], 5).tolist()))
    print(f"  |det| 中位 = {np.median([abs(np.linalg.det(m)) for m in mods]):.6f}")
    print()

    print("=" * 80)
    print("(8) 记忆矩阵: 李群累积, 从后往前铺开")
    print("=" * 80)
    acc, hist = memory_matrix(mods)
    print(f"  累积矩阵 acc = " + str(np.round(acc, 6).tolist()))
    for i in (len(hist) - 1, len(hist) - 2, len(hist) // 2, 1, 0):
        print(f"  hist[{i:>4}] = " + str(np.round(hist[i], 6).tolist()))
    print()

    print("=" * 80)
    print("(9) 位姿矩阵 = 角度 x 记忆矩阵")
    print("=" * 80)
    P = pose_matrix(A, hist)
    print(f"  形状 {P.shape}  (n x 6 x 6)")
    print(f"  P[0] =")
    for row in P[0]:
        print("    [" + " ".join(f"{v:>9.5f}" for v in row) + "]")
    print()
    print(f"  正交性检查 |P[0][:3,:3]^T P[0][:3,:3] - I| 最大 = "
          f"{np.abs(P[0][:3,:3].T @ P[0][:3,:3] - np.eye(3)).max():.3e}")
    print()
    print("=" * 80)
    print("完成。标注 [假设] 的四处需要确认: 第5通道的构成 / 第6个角度 / "
          "方向对齐的细化 / 记忆矩阵乘法顺序")
    print("=" * 80)


if __name__ == "__main__":
    main()
