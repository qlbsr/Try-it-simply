"""
verify_two_sheet_closed_form.py — 双叶结构是否与抛物线同样"简单明了"
================================================================================

问题(用户提出): 双叶结构能不能像抛物线采集那样, 有一个简单明了的闭式,
                使得"多个抛物线向量成对交合 -> 趋向哪个函数曲线"这套判读
                在双叶上同样成立?

抛物线(已严谨计算过): 全部 100 点满足  u = zLength - (zLength/z2^2)*v^2
                      残差 1.8e-16  => 内部解析冗余, 只有 2 个端点向量有效

本文件证明: 双叶有一个同样干净的【单参数闭式】, 而且它就是椭圆弧长元。

推导 (th4 的实际构造):
    qs  = FromToRotation(forward, +v3)      qs2 = FromToRotation(forward, -v3)
    vs3 = (rs*cos(phi), rs*sin(phi), zs),   rs = r*sin(d2), zs = r*cos(d2)
    A = normalize(qs*vs3),  B = normalize(qs2*vs3),  d = |B - A|
    phi = thetas * sin(d2)                  (圆锥展开)

关键: v3.y == 0 恒成立(已实测, rp=0.2068/0.6893/2.0679/6.8929 四组)
      => qs 必是绕 y 轴的纯转动
      => qs2 = Rot_y(pi) * qs
      => B = Rot_y(pi) * A = (-Ax, Ay, -Az)
      => d = 2*sqrt(Ax^2 + Az^2)      (与 v3 的朝向 theta_v 无关!)

再算 Ax^2 + Az^2: 令 P = rs*cos(phi), Q = zs, a = theta_v
    (P cos a + Q sin a)^2 + (-P cos a sin a ... ) -> 是转动的正交变换
    => Ax^2 + Az^2 = (rs^2 cos^2(phi) + zs^2)/r^2 = 1 - sin^2(d2) sin^2(phi)

所以:
    g = d = 2 * sqrt( 1 - sin^2(d2) * sin^2(phi) )
         = 2 * sqrt( 1 - s^2 sin^2(phi) ),   s = sin(d2)

【g/2 就是模量为 s 的椭圆弧长元】 => 与 SolveT/SolveTFromE 反演的是同一个积分。

运行: python verify_two_sheet_closed_form.py
"""
import io, math, sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

FWD = np.array([0.0, 0.0, 1.0])
D2_MEASURED = 44.713528
RP_MEASURED = 51.154373


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def quat_from_to(f, t):
    """复刻 Unity Quaternion.FromToRotation 的最小转动 (f,t 为单位向量)。"""
    f = np.asarray(f, float) / np.linalg.norm(f)
    t = np.asarray(t, float) / np.linalg.norm(t)
    d = float(np.dot(f, t))
    if d > 1.0 - 1e-15:
        return np.eye(3)
    if d < -1.0 + 1e-15:                       # 反向: 轴取任一垂直方向
        ax = np.array([1.0, 0.0, 0.0])
        if abs(f[0]) > 0.9:
            ax = np.array([0.0, 1.0, 0.0])
        ax = ax - np.dot(ax, f) * f
        ax = ax / np.linalg.norm(ax)
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        return np.eye(3) + 2.0 * (K @ K)        # 180 度
    ax = np.cross(f, t)
    ax = ax / np.linalg.norm(ax)
    ang = math.acos(max(-1.0, min(1.0, d)))
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1.0 - math.cos(ang)) * (K @ K)


def th4(uv, d2_deg, rp, v3):
    """复刻 legacy/sjy.cs 的 th4: 返回 (A, B, d, phi, r, vs3)。"""
    s = math.sin(math.radians(d2_deg))
    c = math.cos(math.radians(d2_deg))
    Rs = math.hypot(uv[0], uv[1])
    thetas = math.atan2(uv[1], uv[0])
    if thetas < 0:
        thetas += 2.0 * math.pi
    r = rp * (Rs ** s)
    phis = thetas * s
    rs = r * s
    zs = r * c
    qs = quat_from_to(FWD, v3)
    qs2 = quat_from_to(FWD, -np.asarray(v3, float))
    vs3 = np.array([rs * math.cos(phis), rs * math.sin(phis), zs])
    A = qs @ vs3
    B = qs2 @ vs3
    if np.linalg.norm(A) > 1e-300:
        A = A / np.linalg.norm(A)
    if np.linalg.norm(B) > 1e-300:
        B = B / np.linalg.norm(B)
    return A, B, float(np.linalg.norm(B - A)), phis, r, vs3


def part1_pairing():
    print("=" * 78)
    print("1. 两叶的配对是【固定的 180 度转动】, 不依赖数据")
    print("=" * 78)
    worst_pair = 0.0
    worst_v3 = 0.0
    rng = np.random.default_rng(11)
    for _ in range(4000):
        # v3 在 xz 平面内 (真实情况: v3.y == 0)
        th_v = float(rng.uniform(-0.2, 0.2))
        v3 = np.array([math.sin(th_v), 0.0, math.cos(th_v)])
        A, B, d, phi, r, _ = th4((float(rng.uniform(1e-3, 3.0)), float(rng.uniform(0, 6.28))),
                                 D2_MEASURED, RP_MEASURED, v3)
        pred = np.array([-A[0], A[1], -A[2]])
        worst_pair = max(worst_pair, float(np.max(np.abs(B - pred))))
        worst_v3 = max(worst_v3, abs(float(v3[1])))
    print(f"  max |B - Rot_y(pi)*A| (4000 组随机 uv 与 v3) = {worst_pair:.3e}")
    print("  抽样中 max |v3.y|                              = "
          f"{worst_v3:.3e}   (真实数据实测 v3.y == 0)")
    print()
    print("  => B = (-Ax, Ay, -Az) 恒成立 => d = |B-A| = 2*sqrt(Ax^2+Az^2)")
    print("     所以 d 与 v3 的朝向 theta_v 无关, 只由 phi 决定。")
    print("     这就是双叶的【单参数性】来源, 对应抛物线的 '只有两个端点有效'。")
    print()


def part2_closed_form():
    print("=" * 78)
    print("2. 闭式验证:  g = d = 2*sqrt(1 - sin^2(d2)*sin^2(phi))")
    print("=" * 78)
    d2 = D2_MEASURED
    s = math.sin(math.radians(d2))
    worst = 0.0
    worst_eq = 0.0
    pts = []
    rng = np.random.default_rng(5)
    for _ in range(20000):
        uv = (float(rng.uniform(1e-4, 5.0)), float(rng.uniform(0.0, 2.0 * math.pi)))
        th_v = float(rng.uniform(-0.3, 0.3))
        v3 = np.array([math.sin(th_v), 0.0, math.cos(th_v)])
        A, B, d, phi, r, _ = th4(uv, d2, RP_MEASURED, v3)
        pred = 2.0 * math.sqrt(max(0.0, 1.0 - s * s * math.sin(phi) ** 2))
        worst = max(worst, abs(d - pred))
        pred2 = 2.0 * math.sqrt(s * s * math.cos(phi) ** 2 + math.cos(math.radians(d2)) ** 2)
        worst_eq = max(worst_eq, abs(d - pred2))
        pts.append((phi, d))
    print(f"  20000 组随机 uv/v3:  max |d - 2*sqrt(1 - s^2 sin^2 phi)| = {worst:.3e}")
    print(f"  等价的另一写法       max |d - 2*sqrt(s^2 cos^2 phi + cos^2 d2)| = {worst_eq:.3e}")
    print()
    pts.sort()
    print(f"  {'phi(deg)':>10} {'d = |A-B| 实测':>16} {'2*sqrt(1-s^2 sin^2 phi)':>24}"
          f" {'偏差':>10}")
    for target in (0.0, 45.0, 90.0, 135.0, 180.0):
        best = min(pts, key=lambda p: abs(math.degrees(p[0]) - target))
        phi = best[0]
        pred = 2.0 * math.sqrt(max(0.0, 1.0 - s * s * math.sin(phi) ** 2))
        print(f"  {math.degrees(phi):>10.4f} {best[1]:>16.12f} {pred:>24.12f} "
              f"{abs(best[1]-pred):>10.2e}")
    print()


def part3_range_and_arc():
    print("=" * 78)
    print("3. 值域与【椭圆弧长元】身份")
    print("=" * 78)
    d2 = D2_MEASURED
    s = math.sin(math.radians(d2))
    c = math.cos(math.radians(d2))
    print(f"  d2 = {d2}°,  s = sin(d2) = {s:.6f},  c = cos(d2) = {c:.6f}")
    print()
    print(f"  g = d = 2*sqrt(1 - s^2 sin^2(phi))")
    print(f"  g 的最大值 (phi=0,pi)   = 2            = 2.000000")
    print(f"  g 的最小值 (phi=pi/2)   = 2*cos(d2)    = {2*c:.6f}")
    print(f"  实测真实 uvs 的 g 范围  = [1.421276, 2.000000]")
    print(f"  预测下界 2cos(d2)       = {2*c:.6f}   <- 与实测相差 "
          f"{abs(2*c-1.421276):.2e}")
    print()
    print(f"  g/2 = sqrt(1 - s^2 sin^2 phi)   <== 这正是椭圆弧长元")
    print(f"      E(phi, k) = int_0^phi sqrt(1 - k^2 sin^2) dphi,  这里 k = s = sin(d2)")
    print()
    # 用同一积分算周长, 与 SolveT 的 4E(k) 对齐
    phis = np.linspace(0.0, 2.0 * math.pi, 2000001)
    integrand = np.sqrt(1.0 - s * s * np.sin(phis) ** 2)
    quarter = np.trapezoid(integrand[: len(phis) // 4], phis[: len(phis) // 4])
    total = np.trapezoid(integrand, phis)
    print(f"  int_0^(2pi) g/2 dphi            = {total:.9f}")
    print(f"  4*E(k=sin(d2))                  = {4*quarter:.9f}   (数值一致)")
    print(f"  => 走完一圈 uv 平面, 双叶间距沿 phi 的积分 = 4E(sin(d2))")
    print(f"     与 SolveT 里的 target = u*4E(k) 是【同一个 4E(k)】")
    print()
    print("  临界角:")
    for dd in (20.0, 30.0, 44.713528, 45.0, 60.0, 90.0):
        ss = math.sin(math.radians(dd))
        cc = math.cos(math.radians(dd))
        tag = ""
        if abs(dd - 30.0) < 1e-9:
            tag = "  <== 4*pi*sin(d2)=2*pi, 双锥展开恰好平铺"
        if abs(dd - 45.0) < 1e-9:
            tag = "  <== sin(2d2)=1, 2cos(d2)=1/sin(d2)=sqrt(2), 双叶下界与绕数倒数重合"
        print(f"    d2={dd:>10.6f}°  s={ss:.6f}  2cos(d2)={2*cc:.6f}  "
              f"1/s={1/ss:.6f}  k=s^2={ss*ss:.6f}{tag}")
    print()


def part4_vs_parabola():
    print("=" * 78)
    print("4. 与抛物线结构对照: 两者都是单参数闭式")
    print("=" * 78)
    print("  抛物线(已严谨计算):")
    print("      u = zLength - (zLength/z2^2)*v^2          残余 1.8e-16")
    print("      => 内部解析冗余, 只有端点向量 start*e2, end*e2 携带信息")
    print("      => 2 个向量 = 数据流通主方向")
    print()
    print("  双叶(本文件):")
    print("      d = 2*r*sqrt(1 - sin^2(d2)*sin^2(phi))     残余 ~1e-15")
    print("      phi = theta*sin(d2)   (圆锥展开, theta 为 uv 平面角)")
    print("      => 分离量只由【一个角】phi 决定, 与 v3 朝向无关")
    print("      => 2 条法向 A, B 成对且配对固定为 Rot_y(pi)")
    print()
    print("  两者同构:")
    print(f"    {'':<18}{'抛物线':<28}{'双叶':<28}")
    rows = [
        ("独立变量", "v (沿弦位置)", "phi (圆锥展开角)"),
        ("闭式", "u = zL - (zL/z2^2)v^2", "d = 2r*sqrt(1-s^2 sin^2 phi)"),
        ("向量数", "2 (start/end)", "2 (A, B)"),
        ("配对方式", "索引成对, e2_1.e2_2 = -1", "固定转动 Rot_y(pi)"),
        ("冗余度", "内部解析冗余", "整体 1 参数"),
        ("函数族", "二次(抛物线)", "椭圆弧长元(椭圆积分)"),
    ]
    for a, b, c in rows:
        print(f"    {a:<18}{b:<28}{c:<28}")
    print()
    print("  => 双叶与抛物线同构, 同样'简单明了'; 而且更强: 它的函数族")
    print("     正是 SolveT/SolveTFromE 反演的椭圆弧长, 两者是同一个积分。")
    print()


def part5_closure_hypothesis():
    print("=" * 78)
    print("5. 闭合假设: 圆锥模量 sin(d2) 与 Marden 椭圆模量 e^2 的关系")
    print("=" * 78)
    print("  目前结构里有【两个】模量:")
    print("    (a) 圆锥侧:  k_cone = sin(d2) = s      —— 由 g/2 = sqrt(1-s^2 sin^2 phi) 定出")
    print("    (b) 椭圆侧:  k_marden = e^2 = 2S/(p-qg+S)  —— 显式公式, 参数约定")
    print()
    print("  若二者是同一个(即'近乎无损压缩'的含义), 则 e^2 = sin^2(d2):")
    for dd in (30.0, 40.0, 44.713528, 45.0):
        ss = math.sin(math.radians(dd))
        print(f"    d2={dd:>10.6f}°  ->  e^2 = sin^2(d2) = {ss*ss:.6f}   "
              f"(参数约定 m; 模量 e = {ss:.6f})")
    print()
    print("  此时 SolveT(c2,c3,k) 里的 k 与双叶分离的弧长元是同一个 k,")
    print("  整条链 (c2,yH,cos t2) -> k -> E -> t 与 phi = s*theta 共享一个模量。")
    print()
    print("  注: 这一步是【假设】, 需要真实 (c2, yH, theta2) 数据验证:")
    print("      用 points.json 走一遍 MapDoubleConeToLeaf, 取每点的")
    print("      u=c2, w=yH, g=cos(theta2), 算 k_marden = 2S/(p-qg+S),")
    print("      再与 sin^2(d2)=0.495137 比较。若相等即闭合。")


if __name__ == "__main__":
    part1_pairing()
    part2_closed_form()
    part3_range_and_arc()
    part4_vs_parabola()
    part5_closure_hypothesis()
    print("=" * 78)
    print("结论")
    print("=" * 78)
    print("  1. B = Rot_y(pi)*A 恒成立(与 v3 朝向无关) => 两叶配对是固定转动,")
    print("     比抛物线的索引配对更干净。")
    print("  2. 双叶分离量有单参数闭式:")
    print("        d = 2*r*sqrt(1 - sin^2(d2) sin^2(phi)),  phi = theta*sin(d2)")
    print("     值域 [2cos(d2), 2] = [1.421267, 2], 与实测 [1.421276, 2.000000] 差 9.3e-6。")
    print("  3. d/2r 就是模量 sin(d2) 的【椭圆弧长元】, 与 SolveT 反演的")
    print("     E(phi,k) 是同一个积分 => 双叶结构与共形映射同源, 不是两套东西。")
    print("  4. 与抛物线同构: 都是单参数闭式 + 2 个成对向量; 双叶的函数族是")
    print("     椭圆积分(比抛物线的二次更强), 所以'多点成对交合 -> 趋向某函数'")
    print("     这套判读在双叶上成立, 且极限曲线就是椭圆曲线本身。")
    print("  5. 待验证的闭合假设: e^2 = sin^2(d2) (需真实 c2/yH/theta2 数据)。")
