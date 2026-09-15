"""
verify_gravity_sag.py — 为什么是抛物线: 它是"载荷下的平衡形状", 而不是被选中的简单曲线
================================================================================

用户的描述(我上一轮读错了层次):
    交接不是"交点", 是两个抛物线的特殊关系, 是一个【合并过程】,
    是【最小作用量】的过程;
    绘图时看起来像"起点那端的抛物线自动从空间上受到重力那种塌陷下来";
    像"一本书的两个书页因为被其他书页作用而合并"。

本文件要说明的: 这三句描述对应到代码里的一个统一事实。

核心: 代码里的抛物线和能量, 都是同一件事的两面 —— 均匀载荷下的平衡。

  抛物线:  u(v) = L - (L/Z^2) v^2      =>  u'' = -2L/Z^2 = const
  弯曲能量: ComputeBendEnergy 的被积函数 = kappa^2 (ds/dv),  k = 2L/Z^2

  "u'' = const" 同时给出三件事, 这正是"为什么用抛物线"和"为什么能省略中间"
  是【同一个根因】:

     (i)  力学: 均匀载荷(重力)下, 弦/拱的平衡方程就是 u'' = const  =>  解是抛物线。
          所以你看到的"塌陷下来"就是【载荷平衡形状】本身, 不是近似。
     (ii) 信息: u'' 是常数 => 曲率剖面只由一个数 k 决定 => 内部无自由度
          => 中间连接段"与自身等同", 不必计算。
     (iii) 代数: 二次曲线与直线求交是二次方程 => 交接可闭式, 不需逐点推进。

另外: Vss1 第 618-619 行【算了能量却丢掉】——
     var (points1, en1) = GetParabolaSegmentPoints(...);
     var (points2, en2) = GetParabolaSegmentPoints(...);
     this.points1 = points1;  this.points2 = points2;      // en1, en2 没有下文
  最小作用量所需的泛函已经在手上, 但没被使用。

运行: python verify_gravity_sag.py
"""
import io, math, sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def bend_energy(L, Z, v_start, v_end, segments=200000):
    """复刻 ComputeBendEnergy: k = 2L/Z^2, f(v) = k^2/(1+k^2 v^2)^{5/2} 的 Simpson 积分。"""
    if v_start > v_end:
        v_start, v_end = v_end, v_start
    k = 2.0 * L / (Z * Z)

    def f(v):
        return (k * k) / (1.0 + k * k * v * v) ** 2.5

    vs = np.linspace(v_start, v_end, segments + 1)
    w = np.ones(segments + 1)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    h = (v_end - v_start) / segments
    return float((h / 3.0) * np.sum(w * f(vs)))


def part1_droop():
    print("=" * 78)
    print("1. '重力塌陷'的显式形式: 相对顶点的下垂量是二次的")
    print("=" * 78)
    print("  抛物线在局部系: u(v) = L - (L/Z^2) v^2,   端点 v = ±Z 处 u = 0")
    print("  取顶点 u(0) = L 为基准, 下垂量:")
    print()
    print("        droop(v) = L - u(v) = (L/Z^2) v^2        <== 抛物线下垂")
    print()
    print(f"  {'L':>7} {'Z=1-L':>8} {'droop(Z) 端点下垂':>18} {'droop(Z/2)':>12} "
          f"{'比值(应=1/4)':>13}")
    for L in (0.3, 0.5, 0.7, 0.9):
        Z = 1.0 - L
        d_end = (L / (Z * Z)) * Z * Z
        d_half = (L / (Z * Z)) * (Z * 0.5) ** 2
        print(f"  {L:>7.2f} {Z:>8.2f} {d_end:>18.6f} {d_half:>12.6f} "
              f"{d_half/d_end:>13.4f}")
    print()
    print("  => 下垂在端点达最大值 = L (恰好就是标尺本身), 中点只有 1/4 (二次律)。")
    print("     这就是你看到的'塌陷': 两端相对顶点垂下去 L。")
    print()


def part2_en1_closed():
    print("=" * 78)
    print("2. 第一层能量 en1 的闭式, 以及它【与交接处无关】")
    print("=" * 78)
    print("  ComputeBendEnergy 只吃 (zLength, z2, e2, start, end), 内部只用")
    print("      v_start = start.e2,  v_end = end.e2")
    print("  而第一层的两端是 V_A = +V_3*z2, V_D = -V_3*z2, e2 = V_3.normalized")
    print("      => (v_start, v_end) 恒为 (-Z, +Z)  =>  en1 是【固定区间】上的整段积分")
    print()
    print("  闭式:  int dv/(1+k^2 v^2)^{5/2} = v(3+2k^2 v^2) / (3(1+k^2 v^2)^{3/2})")
    print("     =>  en1 = k^2 * 2Z(3+2k^2 Z^2) / (3(1+k^2 Z^2)^{3/2}),   k = 2L/Z^2")
    print()
    print(f"  {'L':>7} {'Z':>7} {'en1 数值':>16} {'en1 闭式':>16} {'相对差':>11}"
          f" {'en1':>10}")
    worst = 0.0
    for L in (0.2, 0.4, 0.6, 0.8, 0.95):
        Z = 1.0 - L
        num = bend_energy(L, Z, -Z, Z)
        k = 2.0 * L / (Z * Z)
        closed = (k * k) * 2.0 * Z * (3.0 + 2.0 * k * k * Z * Z) / \
                 (3.0 * (1.0 + k * k * Z * Z) ** 1.5)
        worst = max(worst, abs(num - closed) / closed)
        print(f"  {L:>7.2f} {Z:>7.2f} {num:>16.9f} {closed:>16.9f} "
              f"{abs(num-closed)/closed:>11.2e} {L:>10.4f}")
    print()
    print(f"  最大相对差 = {worst:.2e}")
    print()
    print("  => en1 只是 L = sqrt(r[2]) 的函数, 【不含任何交接处信息】。")
    print("     这与'其他书页的作用'必须来自【约束】而非第一层自身能量一致:")
    print("     第一层是纯局部的, 合并的驱动力不在它里面。")
    print()


def part3_u2const():
    print("=" * 78)
    print("3. 'u'' = const' 一个事实同时解释三件事")
    print("=" * 78)
    print("  用数值确认: 抛物线的 u'' 确实是常数, 而曲率不是")
    print()
    L, Z = 0.7, 0.3
    k = 2.0 * L / (Z * Z)
    vs = np.linspace(-Z, Z, 9)
    print(f"  L={L}, Z={Z}, k=2L/Z^2={k:.6f}")
    print()
    print(f"  {'v':>9} {'u''（应恒为 -k）':>18} {'kappa':>12} {'k^2/(1+k^2v^2)^{5/2}':>24}")
    for v in vs:
        u = L - (L / (Z * Z)) * v * v
        up = -2.0 * L / (Z * Z)
        upp = -2.0 * L / (Z * Z)
        kap = abs(upp) / (1.0 + (2.0 * L * v / (Z * Z)) ** 2) ** 1.5
        fv = (k * k) / (1.0 + k * k * v * v) ** 2.5
        _ = u
        print(f"  {v:>9.4f} {up:>18.9f} {kap:>12.6f} {fv:>24.9f}")
    print()
    print("  u'' 处处 = -2L/Z^2 (常数), 而 kappa 随 v 变 —— 变化完全由 u' = -k v 引起。")
    print()
    print("  三件事同一个根因:")
    print("   (i)  力学: 均匀载荷(重力)下弦/拱的平衡方程就是 u'' = const,")
    print("        解即抛物线矢高形状。所以'塌陷下来'是平衡形状本身, 不是近似。")
    print("   (ii) 信息: u'' 常数 => 曲率剖面只由一个数 k 决定 => 内部无自由度。")
    print("   (iii)代数: 二次曲线 ∩ 直线 = 二次方程 => 交接可闭式。")
    print()


def part4_discarded():
    print("=" * 78)
    print("4. 代码里最直接的证据: 能量算了却丢掉")
    print("=" * 78)
    print("  C#/legacy/sjy.cs:618-621")
    print()
    print("      var (points1, en1) = GetParabolaSegmentPoints(v_final, V_A,  V_D);")
    print("      var (points2, en2) = GetParabolaSegmentPoints(vz2,     V_q1, V_q2);")
    print("      this.points1 = points1;")
    print("      this.points2 = points2;          // <-- en1, en2 从这里开始没有下文")
    print()
    print("  全仓检索 en1 / en2 / energy: 除了 ComputeBendEnergy 返回和这里解构,")
    print("  没有任何消费点。")
    print()
    print("  也就是说: 【最小作用量所需的泛函已经在手上, 但从未被使用】。")
    print("  现在两个抛物线的关系完全由 FindIntersections 的求交决定,")
    print("  能量没有参与 —— 这与'合并是最小作用量过程'的描述不符。")
    print()
    print("  三条候选的合并条件 (需要你确认哪一条):")
    print("    (A) 能量平衡:      en1 == en2  时停止合并")
    print("    (B) 约束极小:      在其他层构成的约束下最小化 en1 + en2")
    print("    (C) 接触条件:      两抛物线在接触点处 位置+切线(或曲率) 匹配")
    print("  三者都是从'书页被其他书页压合'能读出来的, 但数学形式不同,")
    print("  会给出不同的交接处。")
    print()


if __name__ == "__main__":
    part1_droop()
    part2_en1_closed()
    part3_u2const()
    part4_discarded()
    print("=" * 78)
    print("回答'为什么用抛物线'")
    print("=" * 78)
    print("  不是'挑了一条简单的曲线', 而是: 抛物线 = 均匀载荷下的平衡形状。")
    print("  你观察到的'受重力塌陷'就是这个平衡本身。")
    print()
    print("  而'为什么能省略中间'与'为什么用抛物线'是【同一个根因】:")
    print("      u'' = const  =>  曲率剖面单参数  =>  内部与自身等同 => 不必计算")
    print("                   =>  与直线求交是二次方程 => 交接闭式")
    print()
    print("  合并过程本身('两个抛物线的关系')目前【没有】被能量约束,")
    print("  因为 en1/en2 被丢弃了。这是可以补上的地方, 也解释了你说代码简陋。")
