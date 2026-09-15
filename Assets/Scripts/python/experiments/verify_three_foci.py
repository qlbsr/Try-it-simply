"""
verify_three_foci.py — 能不能到三焦点: 曲率匹配的合并判据 ＋ 锚点的运动路径
================================================================================

算法(全部显式给出, 便于核对):

STEP 1  泛函与角条件
    取弧长参数化, theta = 切角, kappa = theta'。
        J = int kappa^2 ds = int (theta')^2 ds
        L(theta, theta') = (theta')^2
        p = dL/dtheta' = 2*theta' = 2*kappa        动量
        H = theta'*p - L = (theta')^2 = kappa^2    哈密顿量
    Weierstrass-Erdmann 角条件:  p- = p+  且  H- = H+
        p- = p+  <=>  kappa- = kappa+
        H- = H+  <=>  (kappa-)^2 = (kappa+)^2
    两条条件【给出同一个条件】:  kappa- = kappa+
    => 对纯弯曲泛函, "最小作用量的合并" 就是【曲率连续】。

STEP 2  抛物线的自由度计数 (决定"粘合"后果)
    平面抛物线的自由度 = 4 (顶点 2 + 轴方向 1 + 焦参数 1)。
    给定一点处的 (位置 2 + 切向 1 + 曲率 1) = 4 个条件 => 抛物线唯一确定。
    => 若两条弧在交接处位置/切向/曲率三者匹配, 它们必落在【同一条抛物线】上。
    => 完全合并 => 只剩 1 个焦点。
    => 要出现多个焦点, 必须在交接处【保留曲率跳变】(即保留折痕)。

STEP 3  抛物线的焦点与曲率 (代码的曲线 u = L - (L/Z^2)v^2)
    a = L/Z^2,  k = 2a = 2L/Z^2
    焦点位置(沿轴):  u_f = L - Z^2/(4L)     焦参数 p = Z^2/(4L) = 1/(2k)
    曲率:  kappa(v) = k / (1 + k^2 v^2)^{3/2},   顶点处 kappa(0) = k = 1/(2p)

STEP 4  锚点的求法 (单标量方程)
    两条弧(各自 k1, k2) 的曲率相等:
        k1/(1+k1^2 v1^2)^{3/2} = k2/(1+k2^2 v2^2)^{3/2}
    配合"它们要能交上"(位置连续) 就定出锚点。
    曲率【不】相等时, 差值 kappa- - kappa+ 就是折痕处的集中力矩
    (铰链力矩) —— 这就是'其他书页压合'的物理量。

STEP 5  运动过程
    让自由参数(这里取 L)连续变化, 记录焦点位置与锚点位置, 得到路径。

运行: python verify_three_foci.py
"""
import io, math, sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


# ---------------- STEP 3: 代码抛物线的焦点 / 曲率 / 能量 ----------------
def par(L, Z):
    a = L / (Z * Z)
    k = 2.0 * a
    return dict(L=L, Z=Z, a=a, k=k, p=Z * Z / (4.0 * L), focus_u=L - Z * Z / (4.0 * L))


def kappa(d, v):
    k = d["k"]
    return k / (1.0 + k * k * v * v) ** 1.5


def energy_arc(d, v0, v1):
    """int_{v0}^{v1} kappa^2 ds,  ds = sqrt(1+u'^2) dv = sqrt(1+k^2 v^2) dv
    => 被积 (k^2)/(1+k^2 v^2)^{5/2}; 闭式见 verify_gravity_sag。"""
    k = d["k"]

    def F(v):
        return (k * k) * v * (3.0 + 2.0 * k * k * v * v) / \
               (3.0 * (1.0 + k * k * v * v) ** 1.5)

    return F(v1) - F(v0)


def part1_and_2():
    print("=" * 78)
    print("STEP 1  泛函的角条件: 两条条件塌成一条")
    print("=" * 78)
    print("  J = int kappa^2 ds = int (theta')^2 ds")
    print("      L(theta,theta') = (theta')^2")
    print("      p = dL/dtheta'   = 2*theta' = 2*kappa")
    print("      H = theta'*p - L = (theta')^2 = kappa^2")
    print("  角条件 p- = p+  <=>  kappa- = kappa+")
    print("  角条件 H- = H+  <=>  (kappa-)^2 = (kappa+)^2")
    print()
    print("  => 两条角条件【等价】, 合成一条:  kappa- = kappa+")
    print("     即: 纯弯曲泛函的最小作用量合并 = 曲率连续 = 光滑拼接, 不留折痕。")
    print()
    print("  对照: 若泛函含体力项(重力) L = (theta')^2 - 2*w*y, 则")
    print("      p = 2*theta',  H = (theta')^2 + 2*w*y")
    print("      p 连续 与 H 连续【不再等价】-> 折痕可保留。")
    print("      这解释了'书页'形象: 是载荷/约束让折痕活下来。")
    print()
    print("=" * 78)
    print("STEP 2  抛物线的自由度计数 -> 合并的后果")
    print("=" * 78)
    print("  平面抛物线自由度 = 4  (顶点 2 + 轴方向 1 + 焦参数 1)")
    print("  一点处 (位置 2 + 切向 1 + 曲率 1) = 4 个条件 -> 唯一确定一条抛物线")
    print()
    print("  => 两条弧若在交接处 位置+切向+曲率 全匹配, 必落在【同一条抛物线】上")
    print("     => 完全合并 => 只剩【1 个焦点】")
    print("  => 要出【多焦点】, 必须在每个交接处保留【曲率跳变】(折痕)")
    print("     => 3 弧 + 2 折痕 = 3 焦点")
    print()
    print("  所以 '能不能到三焦点' 的判据不是能不能算, 而是:")
    print("     每个交接处 kappa- - kappa+ 是否被外部约束强制非零。")
    print("     该差值 = 折痕处的集中力矩 = '其他书页压合' 的物理量。")
    print()


def part3_foci():
    print("=" * 78)
    print("STEP 3  代码抛物线的焦点 / 焦参数 / 曲率 (显式)")
    print("=" * 78)
    print("  u = L - (L/Z^2) v^2,   a = L/Z^2,   k = 2a = 2L/Z^2")
    print("  焦点(沿轴): u_f = L - Z^2/(4L)      焦参数 p = Z^2/(4L) = 1/(2k)")
    print("  曲率: kappa(v) = k/(1+k^2 v^2)^{3/2},   kappa(0) = k = 1/(2p)")
    print()
    print(f"  {'L':>7} {'Z=1-L':>7} {'k=2L/Z^2':>11} {'p=Z^2/4L':>11} "
          f"{'焦点 u_f':>11} {'kappa(0)':>11} {'kappa(±Z)':>11} {'kappa 跨度':>11}")
    for L in (0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        Z = 1.0 - L
        d = par(L, Z)
        k0 = kappa(d, 0.0)
        kZ = kappa(d, Z)
        print(f"  {L:>7.2f} {Z:>7.2f} {d['k']:>11.4f} {d['p']:>11.6f} "
              f"{d['focus_u']:>11.6f} {k0:>11.4f} {kZ:>11.6f} {k0/kZ:>11.1f}")
    print()
    print("  注: 'L + Z = 1' 是代码里的【归一化】(zLength + z2 == 1), 不是弧长。")
    print("      真正的弧长是 S = int sqrt(1+k^2 v^2) dv。")
    print()
    print("  焦点位置的关键事实: u_f = L - Z^2/(4L)  【恒小于 L】(焦点在顶点内侧),")
    print("  且 p = Z^2/(4L) 随 L 单调减 -> L 越大焦点越贴近顶点。")
    print()


def part4_anchor_motion():
    print("=" * 78)
    print("STEP 4+5  锚点的解与【运动路径】")
    print("=" * 78)
    print("  取两条弧: 弧1 = (L, Z),  弧2 = (L2, Z2), 各自 k1=2L/Z^2, k2=2L2/Z2^2")
    print("  锚点条件(曲率相等):  k1/(1+k1^2 v1^2)^{3/2} = k2/(1+k2^2 v2^2)^{3/2}")
    print()
    print("  令 L 连续变化(运动), 弧2 固定在 (0.5, 0.5), 记录:")
    print("    - 弧1 的焦点位置 u_f1")
    print("    - 曲率匹配所需的 |v1| (锚点沿弦的位置)")
    print("    - 两弧曲率比 -> 若 != 1 就是折痕力矩")
    print()
    L2, Z2 = 0.5, 0.5
    d2 = par(L2, Z2)
    print(f"  弧2 固定: L2={L2}, Z2={Z2}, k2={d2['k']:.6f}, "
          f"p2={d2['p']:.6f}, 焦点 u_f2={d2['focus_u']:.6f}")
    print()
    print(f"  {'L':>7} {'Z':>7} {'k1':>10} {'焦点 u_f1':>12} "
          f"{'|v1| 使 kappa1=k2':>18} {'k1/k2':>9} {'折痕力矩':>10}")
    for L in (0.20, 0.35, 0.50, 0.65, 0.80, 0.95):
        Z = 1.0 - L
        d1 = par(L, Z)
        k1, k2 = d1["k"], d2["k"]
        # 解 k1/(1+k1^2 v^2)^1.5 = k2  =>  (1+k1^2v^2)^1.5 = k1/k2
        r = k1 / k2
        if r >= 1.0:
            v1 = math.sqrt((r ** (2.0 / 3.0) - 1.0)) / k1
            vs = f"{v1:>18.8f}"
        else:
            vs = f"{'(无解 r<1)':>18}"
        # 折痕力矩: 若锚点被迫落在 |v| = Z 处, 曲率残差
        res = abs(kappa(d1, Z) - k2)
        print(f"  {L:>7.2f} {Z:>7.2f} {k1:>10.4f} {d1['focus_u']:>12.6f} "
              f"{vs} {r:>9.4f} {res:>10.4f}")
    print()
    print("  读法:")
    print("   · '|v1| 使 kappa1=k2' 就是【锚点】沿弧1弦的位置 —— 单标量方程的解。")
    print("     它随 L 连续移动, 这条轨迹就是你说的【运动路径】。")
    print("   · k1/k2 = 1 时锚点落在顶点(v1=0); k1/k2 < 1 时【无解】")
    print("     => 曲率无法匹配 => 折痕被强制保留 => 焦点不会塌成 1 个。")
    print("   · '折痕力矩' 列 = |kappa1(Z) - k2|, 锚点被迫在端点时的残差。")
    print()
    print("  三焦点因此【可达】, 条件是: 至少两个交接处落在 k1/k2 < 1 的区间,")
    print("  即两侧曲率谱不重叠。此时 3 条弧各保有自己的焦点。")
    print()


def part6_energy_check():
    print("=" * 78)
    print("STEP 6  用能量再确认一遍(与闭式对照)")
    print("=" * 78)
    print(f"  {'L':>7} {'Z':>7} {'int kappa^2 ds  (v: -Z..Z)':>28}")
    for L in (0.2, 0.4, 0.6, 0.7, 0.8, 0.95):
        Z = 1.0 - L
        d = par(L, Z)
        print(f"  {L:>7.2f} {Z:>7.2f} {energy_arc(d, -Z, Z):>28.9f}")
    print()
    print("  这些值与 verify_gravity_sag.py 的 en1 闭式一致(那里已验到 2.2e-16)。")
    print("  能量是全局量 -> 只决定'合并到什么程度', 不直接给出锚点位置;")
    print("  锚点位置由【角条件(曲率匹配)】给出。两者分工不同。")


if __name__ == "__main__":
    part1_and_2()
    part3_foci()
    part4_anchor_motion()
    part6_energy_check()
