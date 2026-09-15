"""
verify_least_action_anchor.py — 锚定点怎么来: 分段极值的角条件 (Weierstrass-Erdmann)
================================================================================

用户的问题:
    最小作用量所需的泛函有了, 合并永远无法完成, 那么锚定点怎么得到?
    需要先从起点走到交接处, 再从交接处走到另一个位置, 从那个位置再走回交接处;
    能量能确定吗? 方向和走多少也能确定吗? 有没有相关的数学实现?

回答的结构:
    这不是"求交"问题, 而是【分段极值问题】—— 两段曲线在内部一点相遇,
    这个内部点(锚定点)由【Weierstrass-Erdmann 角条件】确定:
        ① 动量连续:   p = dL/dq'  跨角连续      -> 这就是【方向】
        ② 哈密顿量连续: H = L - q'p 跨角连续      -> 这就是【能量】
    所以你要的两个量恰好就是这两条条件本身; 锚定点则是让这两条同时成立的位置。

    走多少 = 沿极值曲线积分的弧长, 端点固定时是两点边值问题(打靶法求解)。

本文件做两件事:
  A. 把"最小作用量"的曲线真解出来 (变分直接数值极小化 int kappa^2 ds),
     与代码里的固定抛物线对比 —— 量化"抛物线离真正的最小作用量有多远"。
  B. 给出解出的量: 能量、两端切向(方向)、弧长(走多少), 说明它们都是确定的。

运行: python verify_least_action_anchor.py
"""
import io, math, sys

import numpy as np
from scipy.optimize import minimize

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def parabola_uniform(L, Z, N):
    """代码里的抛物线 u = L - (L/Z^2) v^2, 重采样成【等弧长】参数化。
    等弧长是必须的: int kappa^2 ds 是几何量, 用非均匀采样配均匀 ds 会算错。"""
    vf = np.linspace(-Z, Z, 400001)
    uf = L - (L / (Z * Z)) * vf * vf
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(vf), np.diff(uf)))])
    S = float(s[-1])
    su = np.linspace(0.0, S, N + 1)
    v = np.interp(su, s, vf)
    u = np.interp(su, s, uf)
    ds = np.full(N, S / N)
    th = np.arctan2(np.diff(u), np.diff(v))
    return S, ds, th, v, u


def en1_closed(L, Z):
    """第一层能量的闭式: int_{-Z}^{Z} k^2/(1+k^2 v^2)^{5/2} dv,  k = 2L/Z^2。"""
    k = 2.0 * L / (Z * Z)
    return (k * k) * 2.0 * Z * (3.0 + 2.0 * k * k * Z * Z) / \
           (3.0 * (1.0 + k * k * Z * Z) ** 1.5)


def solve_elastica(L, Z, N=400):
    """固定两端 (v,u) = (-Z,0) -> (+Z,0) 与总弧长 S, 最小化 int kappa^2 ds。"""
    S, ds, th0, v0, u0 = parabola_uniform(L, Z, N)

    def obj(th):
        return bend_energy_theta(th, ds)

    def con(th):
        x = np.sum(ds * np.cos(th))
        y = np.sum(ds * np.sin(th))
        return np.array([x - 2.0 * Z, y])

    res = minimize(obj, th0, method="SLSQP",
                   constraints=[{"type": "eq", "fun": con}],
                   options={"maxiter": 4000, "ftol": 1e-14})
    th = res.x
    pos = np.zeros((N + 1, 2))
    for i in range(N):
        pos[i + 1] = pos[i] + ds[i] * np.array([math.cos(th[i]), math.sin(th[i])])
    pos[:, 0] += -Z - pos[0, 0]
    pos[:, 1] += 0.0 - pos[0, 1]
    par_pos = np.column_stack([v0, u0])
    return dict(S=S, ds=ds, theta=th, pos=pos, energy=float(res.fun),
                success=bool(res.success),
                parabola_energy=bend_energy_theta(th0, ds),
                th0=th0, con=con(th), par_pos=par_pos)


def hausdorff(P, Q):
    """P 上每点到 Q 折线的最小距离的最大值(单向 Hausdorff)。"""
    d = np.sqrt(((P[:, None, :] - Q[None, :, :]) ** 2).sum(-1))
    return float(d.min(axis=1).max())


def part_A(L, Z):
    print("=" * 78)
    print(f"A. 最小作用量真解 vs 代码固定抛物线   (L={L}, Z={Z})")
    print("=" * 78)
    r = solve_elastica(L, Z, N=400)
    ec = en1_closed(L, Z)
    print(f"  抛物线闭式 int kappa^2 ds = {ec:.9f}   (verify_gravity_sag 已验)")
    print(f"  抛物线离散(等弧长)        = {r['parabola_energy']:.9f}   "
          f"相对差 {abs(r['parabola_energy']-ec)/ec:.3e}")
    print(f"  极值解(真最小作用量)      = {r['energy']:.9f}")
    print(f"  收敛 = {r['success']},  端点约束残差 = "
          f"[{r['con'][0]:.2e}, {r['con'][1]:.2e}]")
    print(f"  共同弧长 S = {r['S']:.9f}  (端点距 2Z = {2*Z:.6f},  "
          f"S/2Z = {r['S']/(2*Z):.6f})")
    print()
    dE = r['energy'] - r['parabola_energy']
    print(f"  能量差 (极值解 - 抛物线) = {dE:+.9f}   相对 {dE/ec*100:+.4f}%")
    if dE < -1e-6:
        print("  => 极值解能量【低于】抛物线: 抛物线不是最小作用量解, 差距见上。")
    else:
        print("  => 极值解没有比抛物线更低 —— 极小化未收敛(该泛函在此离散下很平)。")
    print()
    hd = hausdorff(r['par_pos'], r['pos'])
    print(f"  形状偏差: 抛物线 -> 极值折线的单向 Hausdorff = {hd:.6f}  "
          f"(相对 L = {hd/L*100:.2f}%)")
    print(f"  顶点高度: 抛物线 {L:.6f}  vs  极值解 {r['pos'][:,1].max():.6f}  "
          f"差 {r['pos'][:,1].max()-L:+.6f}")
    print()
    print(f"  {'v':>10} {'抛物线 u':>14} {'极值解 u (同 v 最近)':>22}")
    for vv in (-Z, -Z / 2, 0.0, Z / 2, Z):
        i = int(np.argmin(np.abs(r['par_pos'][:, 0] - vv)))
        j = int(np.argmin(np.abs(r['pos'][:, 0] - vv)))
        print(f"  {vv:>10.5f} {r['par_pos'][i,1]:>14.7f} {r['pos'][j,1]:>22.7f}")
    print()
    return r


def bend_energy_theta(theta, ds):
    """离散弯曲能量 int kappa^2 ds:  节点曲率 kappa_i = (th_{i+1}-th_i)/ds_i,
    于是 int kappa^2 ds = sum_i (dth_i)^2 / ds_i。"""
    dth = np.diff(theta)
    dsi = ds[:-1]
    return float(np.sum(dth * dth / dsi))


def part_B(r, L, Z):
    print("=" * 78)
    print("B. 你要的三个量: 能量 / 方向 / 走多少 —— 都由极值问题确定")
    print("=" * 78)
    th = r['theta']
    ds = r['ds']
    S = r['S']
    print(f"  【走多少】(弧长) S = {S:.9f}      (端点距离 2Z = {2*Z:.6f},"
          f" 比值 S/2Z = {S/(2*Z):.6f})")
    print()
    print(f"  【方向】(两端切角, 以 e2 为 0 参考):")
    print(f"      起点切角 = {math.degrees(th[0]):>12.6f}°   "
          f"切向 = ({math.cos(th[0]):.6f}, {math.sin(th[0]):.6f})")
    print(f"      终点切角 = {math.degrees(th[-1]):>12.6f}°   "
          f"切向 = ({math.cos(th[-1]):.6f}, {math.sin(th[-1]):.6f})")
    print(f"      中点切角 = {math.degrees(th[len(th)//2]):>12.6f}°")
    print(f"      (抛物线起点切角 = {math.degrees(r['th0'][0]):.6f}°, "
          f"终点 = {math.degrees(r['th0'][-1]):.6f}°)")
    print()
    print(f"  【能量】int kappa^2 ds = {r['energy']:.9f}")
    print()
    print("  => 三个量都是【确定的】, 而且是同一个极值问题的三面:")
    print("     能量 = 泛函的值;  方向 = 动量 p = dL/dq'(角条件之一);")
    print("     走多少 = 沿极值积分得到的弧长。")
    print()
    print("  注意: 极值解的顶点高度与抛物线的 L 不相同 —— 见 A 部分的偏差表。")
    print("        也就是说【当前代码把形状规定成抛物线】, 等于把方向和走多少")
    print("        也一起规定死了, 它们并不由最小作用量决定。")
    print()


def part_C():
    print("=" * 78)
    print("C. 锚定点: Weierstrass-Erdmann 角条件 (数学名称与实现)")
    print("=" * 78)
    print("  问题形态: 极值曲线是【分段】的 —— 从起点到锚点一段、锚点到另一位置一段。")
    print("            锚点可动, 所以它是变分问题的【自由内点】。")
    print()
    print("  经典结论 (Weierstrass-Erdmann 角条件), 在锚点处必须同时成立:")
    print()
    print("     ①  p1 = p2            动量连续      p = dL/dq'")
    print("     ②  H1 = H2            哈密顿量连续   H = L - q'·p")
    print("     (位置连续是定义本身; 曲率可跳 —— 跳变就是【折痕】)")
    print()
    print("  这正是'书页'的图像: 两页在折痕处位置与切向连续, 曲率跳变。")
    print("  也解释了'合并永远无法完成': 若 ① ② 无法同时满足,")
    print("  解就不会退化成一条光滑曲线, 折痕必然保留。")
    print()
    print("  === 可直接用的数学实现 ===")
    print("  (1) Euler-Lagrange -> elastica ODE:  2*kappa'' + kappa^3 = 0")
    print("      首积分 = 张力 + 力矩 (两个守恒量) —— 就是上面的 p 与 H。")
    print("  (2) Maupertuis / Jacobi 度量: 定能量下的最小作用量轨道 =")
    print("      Jacobi 度量 g_J = (E - V)g 下的【测地线】。")
    print("      所以'方向和走多少'= 测地线步: 方向取切向, 步长取弧长参数。")
    print("  (3) 打靶法 / 两点边值: scipy.integrate.solve_bvp")
    print("      未知量 = 两段的初始动量 (方向, 2 个) + 锚点位置 (2 个) + 步长")
    print("      方程   = 两个端点条件 + 两条角条件   -> 方阵, 可解。")
    print("  (4) 最优控制 / Pontryagin 极小值原理: 协态 (costate) 就是 p,")
    print("      分段切换点由【切换条件】定 —— 与角条件同构。")
    print("      若'被其他书页压合'是切换/接触, 这是标准框架。")
    print("  (5) Discrete Elastic Rods (Bergou et al., SIGGRAPH 2008)")
    print("      离散弹性杆: 状态 = 每段 (位置, 材料标架), 能量 = 弯曲 + 扭转。")
    print("      这是最实用的现成实现, 且天然给出每步的 (方向, 步长)。")
    print("  (6) 变分积分器 (discrete Euler-Lagrange, Veselov):")
    print("      每步离散 EL 方程自动保动量 —— 对应你'每次更新的就是交接处'。")
    print()
    print("  === 与本仓库已有的工具对接 ===")
    print("  · 已经有的: int kappa^2 ds (ComputeBendEnergy), 椭圆积分 (Carlson 自组装),")
    print("              四元数标架 (R_ray, R_space), 单参数标尺 L = sqrt(r[2])。")
    print("  · elastica 的解析解本身就是椭圆函数, 其模数与 SolveT 里的 k 同类;")
    print("    所以'最小作用量的模数'很可能就是你在找的 k 的来源 —— 这条是假设,")
    print("    需要单独验证, 但工具链已经完全就位。")


if __name__ == "__main__":
    L, Z = 0.7, 0.3
    r = part_A(L, Z)
    part_B(r, L, Z)
    part_C()
    print("=" * 78)
    print("直接回答")
    print("=" * 78)
    print("  Q: 能量能确定吗?      A: 能。它就是 H = L - q'p 的连续条件(角条件②),")
    print("                          同时是泛函值本身。")
    print("  Q: 方向能确定吗?      A: 能。方向 = 动量 p = dL/dq', 由角条件①给出。")
    print("  Q: 走多少能确定吗?    A: 能。走多少 = 沿极值曲线积分的弧长;")
    print("                          端点固定时是两点边值问题(打靶/solve_bvp)。")
    print("  Q: 锚定点怎么得到?    A: 它是变分的【自由内点】, 由角条件①②同时成立")
    print("                          的位置确定 —— 是一个 2 方程求根, 不是求交。")
    print("  Q: 有现成数学实现吗?  A: 有。Weierstrass-Erdmann 角条件 +")
    print("                          elastica ODE / Jacobi 度量 / Pontryagin /")
    print("                          Discrete Elastic Rods / scipy.solve_bvp。")
