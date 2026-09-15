"""
verify_arc_length_identity.py — 核心关联式 c2/c3 = E(t,k)/(4E(k)) 与圆锥闭合
================================================================================

结构（用户给出）:
    SolveC3:  V = pi*cAxis^2*c3,  cAxis = h1*|w2-w1|/(2pi),
              w1 = sqrt((c3/c2)^2-1) = k1'/k1,  k1 = c2/c3
              w2 = sqrt((c3/h1)^2-1) = k2'/k2,  k2 = h1/c3
    SolveT:   u = c2/c3,  target = u*4E(k),  t = SolveTFromE(target, k)
    核心式:   c2/c3 = E(t,k) / (4E(k))          <-- 本文件验的就是它

    读法: c2/c3 是【归一化弧长】。c2 是点沿椭圆路径的位置读数,
          t 是与之对应的参数角, E 是椭圆的弧长函数。

圆锥闭合:  th4 里 phi = s*theta, s = sin(d2) => 展开平面走一圈 2pi,
          锥面只转 2pi*s。椭圆路径走完一圈 (t: 0->2pi) 弧长为 4E(k),
          锥面方位只推进 2pi*s。闭合需要 s 与绕数匹配。

运行: python verify_arc_length_identity.py
"""
import io, math, sys

import numpy as np
from scipy.special import ellipe, ellipeinc

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

D2_DEG = 44.713528              # 实测 d2
TOL = 1e-15


def solve_t_from_e(target, k, iters=80):
    """完全复刻 C# SolveTFromE: 归约到 [0,4E(k)) 后在 [0,2pi] 上二分。"""
    period = 4.0 * ellipe(k)
    target = target % period
    if target < 0:
        target += period
    lo, hi = 0.0, 2.0 * math.pi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if ellipeinc(mid, k) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def part1_identity():
    print("=" * 74)
    print("1. 核心关联式 c2/c3 = E(t,k)/(4E(k))  —— 恒等式与往返")
    print("=" * 74)
    ks = [0.0, 0.3, 0.6, 0.9, 0.99, 0.999]
    print(f"{'k':>8} {'4E(k) 周长':>12} {'最大往返误差':>14} {'u 单调':>8}")
    for k in ks:
        P = 4.0 * ellipe(k)
        worst = 0.0
        us = []
        for t in np.linspace(0.0, 2.0 * math.pi, 2001):
            u = ellipeinc(t, k) / P
            us.append(u)
            t_back = solve_t_from_e(u * P, k)
            if 0.0 < t < 2.0 * math.pi:      # 端点处归约会退化, 跳过
                worst = max(worst, abs(t_back - t))
        us = np.array(us)
        mono = bool(np.all(np.diff(us) >= -1e-15))
        print(f"{k:>8.3f} {P:>12.6f} {worst:>14.2e} {str(mono):>8}")
    print()
    print("  => E(t,k)/(4E(k)) 把 t∈[0,2pi) 双射到 u∈[0,1), 严格单调。")
    print("     所以 u 天然是【归一化弧长】, SolveTFromE 就是它的反演。")
    print("     端点 t=0 与 t=2pi 都映到 u=0/1 (同一圈的接缝), 归约后不可区分 ——")
    print("     这正是圆锥接缝, 不是数值问题。")
    print()


def part2_cone_closure():
    print("=" * 74)
    print("2. 圆锥闭合: s = sin(d2) 决定展开平面的绕数")
    print("=" * 74)
    s = math.sin(math.radians(D2_DEG))
    print(f"  d2 = {D2_DEG}°  =>  s = sin(d2) = {s:.6f}")
    print()
    print(f"  一个锥面 (单叶) 展开      : 2pi*s = {math.degrees(2*math.pi*s):9.4f}°")
    print(f"  双叶 (双锥) 展开总角      : 4pi*s = {math.degrees(4*math.pi*s):9.4f}°")
    print(f"  与整圈 360° 的差 (超出量) : {math.degrees(2*math.pi*(2*s-1)):9.4f}°")
    print()
    if 2 * s > 1.0:
        print("  => 4pi*s > 2pi: 双锥展开【超出】一整圈 —— 单平面放不下,")
        print("     必须多层/带分支割线。这就是椭圆函数(双周期)进场的理由。")
    print()
    print(f"  椭圆路径走完一圈 (t:0->2pi, 弧长 4E(k)) 时, 锥面方位只推进 2pi*s")
    print(f"  => 走满整圈锥面需要 1/s = {1.0/s:.6f} 次路径遍历")
    print(f"     即每次遍历留下闭合缺陷 2pi(1-s) = {math.degrees(2*math.pi*(1-s)):.4f}°")
    print()
    print("  闭合条件: 存在整数 n 使 n*s ∈ Z (或 n*(1-s) ∈ Z)")
    best = []
    for n in range(1, 61):
        frac = (n * s) % 1.0
        err = min(frac, 1.0 - frac)
        best.append((err / n, err, n))
    best.sort()
    print(f"  {'分母 n':>8} {'n*s 距最近整数的距离':>22}")
    for _, err, n in best[:6]:
        print(f"  {n:>8} {err:>22.6f}")
    print()
    print("  => 最优也只是近似; d2=44.713528° 下 s 不是低阶有理数,")
    print("     这是结构【不闭合】的定量证据。")
    print()
    print("  临界角: 4pi*s = 2pi  <=>  s = 1/2  <=>  d2 = 30°")
    for d in (20.0, 30.0, 44.713528, 60.0, 90.0):
        ss = math.sin(math.radians(d))
        tag = ""
        if abs(2 * ss - 1.0) < 1e-12:
            tag = "   <== 展开恰好平铺 (4pi*s = 2pi)"
        print(f"    d2={d:>10.6f}°  s={ss:.6f}  4pi*s={math.degrees(4*math.pi*ss):9.3f}°{tag}")
    print()


def part3_two_moduli():
    print("=" * 74)
    print("3. c2 与 h1 各自带一个模量, cAxis 量的是两者的失配")
    print("=" * 74)
    # 真实的 c2/h1/c3 尺度 (从 rp/d2 几何取一组代表性值)
    rp = 51.154373
    alpha = math.radians(D2_DEG)
    h, H = rp * math.cos(alpha), math.pi * rp * math.sin(alpha)
    print(f"  rp={rp:.6f}  h={h:.6f}  H={H:.6f}")
    print()
    print(f"  {'c3/c2':>9} {'k1=c2/c3':>10} {'w1=k1p/k1':>12} "
          f"{'k2=h1/c3':>10} {'w2=k2p/k2':>12} {'|w2-w1|':>10}")
    for ratio in (1.05, 1.2, 1.5, 2.0, 3.0, 5.0):
        c2 = 1.0
        c3 = c2 * ratio
        h1 = c2 * 0.8                       # h1 与 c2 同量级
        w1 = math.sqrt((c3 / c2) ** 2 - 1.0)
        w2 = math.sqrt((c3 / h1) ** 2 - 1.0)
        k1, k2 = c2 / c3, h1 / c3
        print(f"  {ratio:>9.2f} {k1:>10.6f} {w1:>12.6f} "
              f"{k2:>10.6f} {w2:>12.6f} {abs(w2-w1):>10.6f}")
    print()
    print("  w1 = sqrt((c3/c2)^2-1) = sqrt(1/k1^2-1) = k1'/k1   (恒等)")
    print("  => cAxis = h1*|w2-w1|/(2pi) 是【两个环模的失配量】乘 h1/(2pi)。")
    print("     1/(2pi) 是绕数归一 —— 失配量是角度性质的量, 不是长度。")
    print()
    for ratio in (1.05, 1.2, 1.5, 2.0):
        c2 = 1.0
        c3 = ratio
        h1 = 0.8
        w1 = math.sqrt((c3 / c2) ** 2 - 1.0)
        w2 = math.sqrt((c3 / h1) ** 2 - 1.0)
        k1 = c2 / c3
        lhs = math.sqrt((c3 / c2) ** 2 - 1.0)
        rhs = math.sqrt(1.0 / k1 ** 2 - 1.0)
        print(f"    c3/c2={ratio:.2f}: |w1 - k1'/k1| = {abs(lhs-rhs):.3e}")
    print()


def part4_monotonicity():
    print("=" * 74)
    print("4. SolveC3 的二分前提: F(c3) 真的单调递增吗?")
    print("=" * 74)
    print("  F(c3) = pi*cAxis^2*c3 - V ,  cAxis = h1*|w2-w1|/(2pi)")
    print("       => F(c3) = h1^2*(w2-w1)^2*c3/(4pi) - V")
    print("       => F 单调递增  <=>  Q(c3) = (w2-w1)^2 * c3 单调递增")
    print()
    hdr = (f"  {'c2':>5} {'h1':>5} {'lo':>6} "
           f"{'Q 最小处 c3/lo':>15} {'Q(lo+)':>10} {'Q(最小)':>10} {'单调?':>7} {'根数':>5}")
    print(hdr)
    for c2, h1 in [(1.0, 0.95), (1.0, 0.8), (1.0, 0.5),
                   (1.0, 0.2), (1.0, 0.05), (2.0, 0.3)]:
        lo = max(c2, h1)
        cs = np.linspace(lo * (1 + 1e-9), lo * 25.0, 500001)

        def Q(c):
            w1 = np.sqrt((c / c2) ** 2 - 1.0)
            w2 = np.sqrt((c / h1) ** 2 - 1.0)
            return (w2 - w1) ** 2 * c

        q = Q(cs)
        i = int(np.argmin(q))
        mono = bool(np.all(np.diff(q) >= -1e-12))
        # 计数: Q 的局部极小个数 = 单调性破坏的段数
        dq = np.diff(q)
        sign = np.sign(dq)
        turns = int(np.sum((sign[1:] > 0) & (sign[:-1] < 0)))
        print(f"  {c2:>5.2f} {h1:>5.2f} {lo:>6.2f} "
              f"{cs[i]/lo:>15.4f} {q[0]:>10.5f} {q[i]:>10.5f} "
              f"{str(mono):>7} {turns + 1:>5}")
    print()
    print("  => 注意: 所有取样行【全部】非单调 (连 h1/c2=0.95 也是如此),")
    print("     不是边缘情形, 而是只要 h1<c2 就发生。")
    print("     F 出现【局部极小 → 两个根】, 而 SolveC3 的二分")
    print("     ('F(mid)>0 就收 hi') 建立在单调假设上, 会收敛到其中一个根")
    print("     (不保证是预期根), 且 hi 倍增阶段可能直接跳过负值区间。")
    print()
    print("  最小值位置 (h1/c2 -> 首个极小出现的 c3/c2):")
    for h1 in (0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2):
        c2 = 1.0
        cs = np.linspace(1.0 * (1 + 1e-9), 25.0, 500001)
        w1 = np.sqrt(cs ** 2 - 1.0)
        w2 = np.sqrt((cs / h1) ** 2 - 1.0)
        q = (w2 - w1) ** 2 * cs
        i = int(np.argmin(q))
        print(f"    h1/c2={h1:.2f}  极小在 c3/c2 = {cs[i]:.4f}   "
              f"Q 谷底/起点 = {q[i]/q[0]:.5f}")
    print()


if __name__ == "__main__":
    part1_identity()
    part2_cone_closure()
    part3_two_moduli()
    part4_monotonicity()
    print("=" * 74)
    print("结论")
    print("=" * 74)
    print("  1. c2/c3 = E(t,k)/(4E(k)) 是 SolveT 的构造式 => 链是【弧长驱动】:")
    print("     给定 k 后, c2/c3 就是归一化弧长, t 是与之对应的均匀参数角。")
    print("     k 本身仍是外部输入 (SolveT 的第三个参数), 来源未定。")
    print("  2. c2 是点沿路径的弧长位置读数; 这就是'从 c2 + 体积比还原点集信息'。")
    print("     c2/c3∈(0,1) 由 SolveC3 的约束 c3>max(c2,h1) 自动保证。")
    print("  3. 闭合由 s=sin(d2) 控制: 路径一圈 (弧长 4E(k)) 只推锥面 2pi*s,")
    print("     需 1/s 次遍历才满圈; 每次留下缺陷 2pi(1-s)。")
    print("  4. d2>30° => 4pi*s>2pi => 双锥展开超出整圈 (d2=44.71° 时超出 146.6°)")
    print("     => 单平面放不下, 必须多值/椭圆函数。d2=30° 是展开恰好平铺的临界角。")
    print("  5. cAxis = h1*|w2-w1|/(2pi), 其中 w1 = k1'/k1 (已验证恒等, 误差 0~2.2e-16)")
    print("     => cAxis 是【两个环模的失配量】, 1/(2pi) 是绕数归一, 是角度性质的量。")
    print("  6. F(c3) 【非单调】: 只要 h1<c2, Q=(w2-w1)^2*c3 就存在极小")
    print("     (极小在 c3/c2 = 1.006~1.260, 随 h1/c2 减小而贴近 1),")
    print("     => F 可有两个根。SolveC3 的二分('F(mid)>0 就收 hi')不保证")
    print("        收敛到预期根; 且初始括号 hi=2*lo 恰好把两个根都包进来。")
    print("        修法: 先定位 Q 的谷底, 只在单侧单调分支上二分。")
    print("  7. 活路径把 u 写成被平方的 opi、把角度写成体积比 vabc/Vc,")
    print("     丢掉了第 1/2 条的弧长均匀性 —— 这是真实 uvs 跨 3.94 dex 的根源。")
