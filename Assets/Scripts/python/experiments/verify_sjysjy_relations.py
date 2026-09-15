"""
verify_sjysjy_relations.py
--------------------------
分析 C#\sjysjy.cs 里被"浓缩"成关联式的部分:

  MardenEllipse.SolveC3 / F      (line 115 / 144)   <- 新增的关联式与它的反解
  MardenEllipse.SolveT / SolveTFromE (line 156 / 170)
  MardenEllipse.th2 / dd2        (line 198 / 209)
  MapDoubleConeToLeaf            (line 253)

核心关联式 (从代码里提取):
    w1 = sqrt((c3/c2)^2 - 1)
    w2 = sqrt((c3/h1)^2 - 1)
    cAxis = h1*|w2 - w1| / (2*pi)
    V     = pi * cAxis^2 * c3
等价闭式:
    V = c3/(4*pi) * [ sqrt(c3^2 - h1^2) - (h1/c2)*sqrt(c3^2 - c2^2) ]^2
"""
import numpy as np
from scipy.special import ellipe, ellipeinc

F = np.float64


# ------------------------------------------------------- 代码原式
def V_code(c3, c2, h1):
    w1 = np.sqrt((c3 / c2) ** 2 - 1.0)
    w2 = np.sqrt((c3 / h1) ** 2 - 1.0)
    cAxis = np.abs(h1 * w2 - h1 * w1) / (2.0 * np.pi)
    return np.pi * cAxis * cAxis * c3


# ------------------------------------------------------- 等价闭式
def V_closed(c3, c2, h1):
    return c3 / (4.0 * np.pi) * (np.sqrt(c3 ** 2 - h1 ** 2)
                                 - (h1 / c2) * np.sqrt(c3 ** 2 - c2 ** 2)) ** 2


def F_res(c3, c2, h1, V):
    """C# F(): 返回 Vcalc - V"""
    if c3 <= max(c2, h1):
        return -1e18
    return V_code(c3, c2, h1) - V


def solve_c3(c2, h1, V, verbose=False):
    """C# SolveC3() 的忠实复刻"""
    if V <= 0:
        return max(c2, h1) + 1e-9
    if c2 <= 0:
        c2 = 1e-12
    if h1 <= 0:
        h1 = 1e-12
    lo = max(c2, h1) + 1e-9
    hi = lo * 2.0
    if F_res(lo, c2, h1, V) > 0:
        return ("CLAMP", lo) if verbose else lo
    for _ in range(200):
        if F_res(hi, c2, h1, V) > 0:
            break
        hi *= 2.0
        if hi > 1e15:
            break
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        if F_res(mid, c2, h1, V) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


if __name__ == "__main__":
    print("=" * 80)
    print("1. 关联式: 代码 V_code 与闭式 V_closed 是否一致 ?")
    print("=" * 80)
    rng = np.random.default_rng(5)
    worst = 0.0
    for _ in range(20000):
        c2 = 10 ** rng.uniform(-2, 2)
        h1 = 10 ** rng.uniform(-2, 2)
        c3 = max(c2, h1) * (1 + 10 ** rng.uniform(-3, 2))
        a, b = V_code(c3, c2, h1), V_closed(c3, c2, h1)
        worst = max(worst, abs(a - b) / max(abs(a), 1e-300))
    print(f"  20000 组随机 (c2,h1,c3): 最大相对差 = {worst:.3e}")
    print("  => V = c3/(4pi) * [ sqrt(c3^2-h1^2) - (h1/c2)*sqrt(c3^2-c2^2) ]^2")

    print()
    print("=" * 80)
    print("2. c2 == h1 时 V 恒等于 0 ?  (w1 == w2 => cAxis == 0)")
    print("=" * 80)
    h1 = 0.8
    for c3 in [1.0, 2.0, 10.0, 1e3]:
        print(f"    c2 = h1 = {h1}   c3 = {c3:<8g}  V = {V_code(c3, h1, h1):.3e}")
    print("  => c2 = h1 时 cAxis 恒为 0, V 恒为 0。此时 SolveC3 无解 (任何 c3 都给 V=0)。")

    print()
    print("=" * 80)
    print("3. V(c3) 在 c3 > max(c2,h1) 上单调吗 ? (SolveC3 的二分需要 F 单调)")
    print("=" * 80)
    print(f"  {'c2':>10} {'h1':>10} {'c2/h1':>8} {'V(lo)':>14} {'V(10lo)':>14} "
          f"{'dV/dc3 变号次数':>16} {'V 的最小值处':>14}")
    tests = [(0.0222, 3.693), (25.72, 3.693), (1.0, 1.0), (2.0, 1.0), (1.0, 2.0),
             (0.5, 0.5), (3.0, 0.1), (0.1, 3.0)]
    for c2, h1 in tests:
        lo = max(c2, h1)
        c3s = np.linspace(lo * (1 + 1e-9), lo * 100, 200000)
        Vs = V_closed(c3s, c2, h1)
        dV = np.diff(Vs)
        sign_changes = int(np.sum(np.diff(np.sign(dV)) != 0))
        imin = int(np.argmin(Vs))
        print(f"  {c2:>10.4g} {h1:>10.4g} {c2/h1:>8.4f} {Vs[0]:>14.5e} {Vs[-1]:>14.5e} "
              f"{sign_changes:>16} {c3s[imin]/lo:>14.6f} lo")

    print()
    print("=" * 80)
    print("4. SolveC3 往返: c3 -> V -> SolveC3 -> c3'")
    print("=" * 80)
    print(f"  {'c2':>10} {'h1':>10} {'c3_true':>12} {'c3_back':>14} {'相对误差':>12} {'状态':>9}")
    for c2, h1 in [(0.0222, 3.693), (25.72, 3.693), (2.0, 1.0), (1.0, 2.0), (3.0, 0.1)]:
        for fac in [1.05, 2.0, 20.0]:
            c3t = max(c2, h1) * fac
            V = V_code(c3t, c2, h1)
            r = solve_c3(c2, h1, V, verbose=True)
            if isinstance(r, tuple):
                st, val = r
                print(f"  {c2:>10.4g} {h1:>10.4g} {c3t:>12.5g} {val:>14.5g} "
                      f"{'-':>12} {st:>9}")
            else:
                print(f"  {c2:>10.4g} {h1:>10.4g} {c3t:>12.5g} {r:>14.5g} "
                      f"{abs(r-c3t)/c3t:>12.3e} {'OK':>9}")

    print()
    print("=" * 80)
    print("5. V 低于可达下界时 SolveC3 会静默夹住")
    print("=" * 80)
    c2, h1 = 2.0, 1.0
    lo = max(c2, h1)
    Vmin = V_closed(np.array([lo * (1 + 1e-9)]), c2, h1)[0]
    print(f"  c2={c2}, h1={h1}: c3 -> lo 时 V -> {Vmin:.6e}")
    for Vin in [Vmin * 0.5, Vmin * 1e-3, -1.0]:
        r = solve_c3(c2, h1, Vin, verbose=True)
        print(f"    请求 V = {Vin:.4e}  ->  {r}   (F(lo) = {F_res(lo+1e-9,c2,h1,Vin):.4e})")
    print("  => 返回 lo = max(c2,h1)+1e-9, 但 F(lo) > 0 说明这不是根 —— 无警告。")

    print()
    print("=" * 80)
    print("6. SolveTFromE: 解的是 E(phi,k) = target (第二类不完全椭圆积分的反函数)")
    print("=" * 80)
    def solve_t_from_e(target, k, iters=80):
        Ef = ellipe(k)
        period = 4.0 * Ef
        target = target % period
        if target < 0:
            target += period
        lo, hi = 0.0, 2.0 * np.pi
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if ellipeinc(mid, k) < target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    print(f"  {'k':>8} {'phi_true':>10} {'E(phi,k)':>14} {'phi_back':>12} {'误差':>12} "
          f"{'dE/dphi':>12}")
    for k in [0.0, 0.3, 0.7, 0.9, 0.99, 0.9999, 1 - 1e-12]:
        for phi_t in [0.3, 1.2, 2.5, 5.0]:
            E = ellipeinc(phi_t, k)
            pb = solve_t_from_e(E, k)
            dEdphi = np.sqrt(1 - k * k * np.sin(pb) ** 2)
            print(f"  {k:>8.6g} {phi_t:>10.4f} {E:>14.8f} {pb:>12.8f} {abs(pb-phi_t):>12.2e} "
                  f"{dEdphi:>12.3e}")
    print("  E(2pi,k) = 4E(k) = 完整周长 (单位长半轴, 离心率 k)  => 二分区间 [0,2pi] 正确")
    print("  k -> 1 时 phi 靠近 pi/2 处 dE/dphi -> 0, 反解病态 (下面单独量化)。")

    print()
    print("=" * 80)
    print("7. k -> 1 时反解的病态 (float32 输入的话)")
    print("=" * 80)
    for k in [0.9, 0.99, 0.999, 1 - 1e-4, 1 - 1e-6, 1 - 1e-12]:
        # 在 phi = pi/2 附近 dE/dphi 最小
        dmin = np.sqrt(max(1 - k * k, 0.0))
        # float32 下 c2/c3 的相对误差 ~ 6e-8, target = u*4E(k)
        du = 6e-8
        dtarget = du * 4 * ellipe(k)
        dphi = dtarget / max(dmin, 1e-300)
        print(f"  k = {k:<12.10g}  min dE/dphi = {dmin:.3e}   "
              f"若 u=c2/c3 只有 float32 精度 -> dphi ~ {dphi:.3e} rad")
    print("  => k 越接近 1, arc-length 反解对输入精度越敏感。")
