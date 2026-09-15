"""
verify_junction_path.py — 交接处的路径: 折线与抛物线的交点闭式
================================================================================

用户要的: "各个空间内实体向量接近交接处的路径"。
只要知道这条路径, 就不必算中间复杂的交接与变换过程。

代码证据 (legacy/sjy.cs:567-733, Vss1 + FindIntersections):

    zLength = sqrt(r[2])          <-- r[2] 只提供一个【合理的长度】
    z2      = 1 - zLength         <-- 注意: zLength + z2 == 1 (归一化)
    V_3     = cross(v_final, v_2).normalized          (同时垂直于两者)
    V_A     = +V_3 * z2                               |V_A| = z2
    V_D     = -V_3 * z2
    v_c     = v_final - V_A - v_final.normalized*z2
    v_c1    = v_final + V_A

    FindIntersections(v_final, V_A, V_3, v_c, v_c1):
        把 v_c, v_c1 投影到局部系 (e1 = v_final 方向, e2 = V_3)
        解  A*s^2 + B*s + C = 0  —— 即【弦 与 抛物线 u = zLength - (zLength/z2^2)v^2 的交点】
        A = (zLength/z2^2)*dv^2
        B = (zLength/z2^2)*2*v0*dv + du
        C = (zLength/z2^2)*v0^2 + u0 - zLength
        只保留 s ∈ [0,1] 的根; 不足两个则降级返回 (v_c, v_c1)

本文件证明:
  1. v_c, v_c1 在 (e1, e2) 系的投影是【常数形式】, 与 v_2 / v_3 的朝向无关:
         (u0, v0) = (L - Z, -Z),   (u1, v1) = (L, Z),    L = zLength, Z = z2
  2. 于是二次方程系数与判别式全部只有 L 一个自变量:
         A = 4L,  B = 1 - 5L,  C = 2L - 1,  D = (1 - L)(1 + 7L)
     交接点参数:  s± = [ 5L - 1 ± sqrt((1-L)(1+7L)) ] / (8L)
  3. D > 0 恒成立 => 两个实交【永远存在】;
     但 s ∈ [0,1] 过滤在 L < 0.5 时只剩一个根 => 走【降级分支】。
     L = 0.5 (即 r[2] = 0.25) 是行为分界的临界点。
  4. 因此: 整个交接几何是【单参数 L = sqrt(r[2]) 的函数】。
     这就是"r[2] 只提供合理长度"的准确含义 —— 它不是数据, 是标尺。

运行: python verify_junction_path.py
"""
import io, math, sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def build(L, theta_v2):
    """按 Vss1 构造 v_final 与交接所需向量。L = zLength, Z = 1 - L。"""
    Z = 1.0 - L
    v_final = np.array([0.0, 0.0, L])
    v_2 = np.array([math.sin(theta_v2), 0.0, math.cos(theta_v2)])
    V_3 = np.cross(v_final, v_2)
    n = np.linalg.norm(V_3)
    V_3 = V_3 / n if n > 1e-300 else np.array([0.0, 1.0, 0.0])
    V_A = V_3 * Z
    V_D = -V_3 * Z
    e1 = v_final / np.linalg.norm(v_final)
    v_c = v_final - V_A - e1 * Z
    v_c1 = v_final + V_A
    return dict(L=L, Z=Z, v_final=v_final, v_2=v_2, V_3=V_3, V_A=V_A, V_D=V_D,
                e1=e1, e2=V_3, v_c=v_c, v_c1=v_c1)


def find_intersections(d):
    """复刻 FindIntersections。"""
    L, Z = d["L"], d["Z"]
    e1, e2 = d["e1"], d["e2"]
    u0, v0 = float(np.dot(d["v_c"], e1)), float(np.dot(d["v_c"], e2))
    u1, v1 = float(np.dot(d["v_c1"], e1)), float(np.dot(d["v_c1"], e2))
    du, dv = u1 - u0, v1 - v0
    A = (L / (Z * Z)) * dv * dv
    B = (L / (Z * Z)) * 2.0 * v0 * dv + du
    C = (L / (Z * Z)) * v0 * v0 + u0 - L
    disc = B * B - 4.0 * A * C
    sVals = []
    if disc >= 0:
        sq = math.sqrt(disc)
        for s in ((-B - sq) / (2 * A), (-B + sq) / (2 * A)):
            if 0.0 <= s <= 1.0:
                sVals.append(s)
        sVals.sort()
    if len(sVals) >= 2:
        return (d["v_c"] + sVals[0] * (d["v_c1"] - d["v_c"]),
                d["v_c"] + sVals[1] * (d["v_c1"] - d["v_c"]),
                sVals, (u0, v0, u1, v1), (A, B, C, disc))
    return (d["v_c"], d["v_c1"], sVals, (u0, v0, u1, v1), (A, B, C, disc))


def part1_projections():
    print("=" * 78)
    print("1. 弦两端在 (e1, e2) 系的投影是常数形式, 与 v_2 / v_3 朝向无关")
    print("=" * 78)
    print("  断言: (u0, v0) = (L - Z, -Z)   (u1, v1) = (L, Z)")
    print()
    w_u = w_v = 0.0
    for L in np.linspace(0.05, 0.95, 37):
        for th in np.linspace(0.15, math.pi - 0.15, 25):
            d = build(float(L), float(th))
            _, _, _, (u0, v0, u1, v1), _ = find_intersections(d)
            Z = 1.0 - L
            w_u = max(w_u, abs(u0 - (L - Z)), abs(u1 - L))
            w_v = max(w_v, abs(v0 - (-Z)), abs(v1 - Z))
    print(f"  925 组 (L, theta_v2):  max |u 偏差| = {w_u:.3e}   max |v 偏差| = {w_v:.3e}")
    print()
    print("  => 弦的局部坐标【完全由 L 决定】, v_2 的朝向不进入。")
    print()


def part2_coeffs():
    print("=" * 78)
    print("2. 二次方程系数与交点闭式: 只有 L 一个自变量")
    print("=" * 78)
    print("  推导:  Z = 1 - L,  du = Z,  dv = 2Z")
    print("      A = (L/Z^2)*dv^2         = 4L")
    print("      B = (L/Z^2)*2*v0*dv + du = -4L + Z = 1 - 5L")
    print("      C = (L/Z^2)*v0^2 + u0 - L = L - Z  = 2L - 1")
    print("      D = B^2 - 4AC = (1-5L)^2 - 16L(2L-1) = 1 + 6L - 7L^2 = (1-L)(1+7L)")
    print()
    wa = wb = wc = wd = 0.0
    for L in np.linspace(0.02, 0.98, 97):
        d = build(float(L), 1.0)
        _, _, _, _, (A, B, C, disc) = find_intersections(d)
        wa = max(wa, abs(A - 4 * L))
        wb = max(wb, abs(B - (1 - 5 * L)))
        wc = max(wc, abs(C - (2 * L - 1)))
        wd = max(wd, abs(disc - (1 - L) * (1 + 7 * L)))
    print(f"  97 组 L:  max|A-4L|={wa:.3e}  max|B-(1-5L)|={wb:.3e}  "
          f"max|C-(2L-1)|={wc:.3e}  max|D-(1-L)(1+7L)|={wd:.3e}")
    print()
    print("  交接点参数闭式:   s± = [ 5L - 1 ± sqrt((1-L)(1+7L)) ] / (8L)")
    print()
    print(f"  {'L':>8} {'r[2]=L^2':>10} {'Z=1-L':>8} {'s-':>12} {'s+':>12} "
          f"{'交点个数':>9} {'分支':>10}")
    ws = 0.0
    for L in (0.15, 0.30, 0.45, 0.50, 0.55, 0.70, 0.85, 0.95):
        d = build(L, 1.0)
        _, _, sVals, _, (A, B, C, disc) = find_intersections(d)
        D = (1 - L) * (1 + 7 * L)
        sm = (5 * L - 1 - math.sqrt(D)) / (8 * L)
        sp = (5 * L - 1 + math.sqrt(D)) / (8 * L)
        ws = max(ws, abs(sp - ((5 * L - 1 + math.sqrt(D)) / (8 * L))))
        branch = "正常(双交)" if len(sVals) >= 2 else "降级(端点)"
        sm_s = f"{sm:>12.6f}" if len(sVals) >= 2 else f"{sm:>12.6f}*"
        print(f"  {L:>8.2f} {L*L:>10.4f} {1-L:>8.2f} {sm_s:>12} {sp:>12.6f} "
              f"{len(sVals):>9} {branch:>10}")
    print("  (* 表示该根落在 [0,1] 之外, 被过滤掉)")
    print()


def part3_regime():
    print("=" * 78)
    print("3. 行为分界: L = 0.5  (即 r[2] = 0.25)")
    print("=" * 78)
    print("  s- = 0  <=>  5L - 1 = sqrt((1-L)(1+7L))")
    print("          <=>  25L^2 - 10L + 1 = 1 + 6L - 7L^2")
    print("          <=>  32L^2 - 16L = 0  <=>  L(32L - 16) = 0  <=>  L = 0.5")
    print()
    print(f"  {'L':>8} {'s-':>12} {'s+':>12} {'s- 在[0,1]?':>13} {'走哪个分支':>14}")
    for L in (0.10, 0.25, 0.40, 0.4999, 0.5001, 0.60, 0.80, 0.94):
        D = (1 - L) * (1 + 7 * L)
        sm = (5 * L - 1 - math.sqrt(D)) / (8 * L)
        sp = (5 * L - 1 + math.sqrt(D)) / (8 * L)
        ok = (0.0 <= sm <= 1.0)
        print(f"  {L:>8.4f} {sm:>12.6f} {sp:>12.6f} {str(ok):>13} "
              f"{'正常(双交)' if ok else '降级(端点)':>14}")
    print()
    print("  => D > 0 恒成立(两个实根永远存在), 但 s ∈ [0,1] 过滤使")
    print("     L < 0.5 时只剩 s+ 一个根 -> sVals.Count == 1 -> 走 else 降级分支,")
    print("     返回 (v_c, v_c1) 即弦的两端, 抛物线交点被【丢弃】。")
    print("     这是一个静默的 regime 切换, 由 r[2] 是否 > 0.25 决定。")
    print()


def part4_path():
    print("=" * 78)
    print("4. 结论: '接近交接处的路径' 的显式形式")
    print("=" * 78)
    print("  路径由两条曲线组成, 交接 = 两者相交:")
    print()
    print("    空间路径(曲线):  抛物线   u = L - (L/Z^2) v^2")
    print("    折线路径(直线):  弦       (u,v) = (L-Z,-Z) -> (L,Z)")
    print("    交接处(交点):    s± = [5L-1 ± sqrt((1-L)(1+7L))]/(8L)")
    print("                     交点 = v_c + s * (v_c1 - v_c)")
    print()
    print("  而 L = sqrt(r[2]), Z = 1 - L  => 全链条只有 r[2] 一个输入。")
    print()
    print("  所以'不用算中间复杂的交接和变换过程'是成立的, 而且原因有三层:")
    print("    (a) 抛物线 u'' = const  => 曲率剖面 kappa(v) = k/(1+k^2 v^2)^{3/2}")
    print("        只由一个数 k = 2L/Z^2 决定, 内部没有任何自由度;")
    print("    (b) 能量 int kappa^2 ds 的被积函数只吃 v = 点.e2,")
    print("        整条弧的能量由两端投影 (v_start, v_end) 决定;")
    print("    (c) 交接点是【直线 与 二次曲线 求交】= 二次方程,")
    print("        闭式即可给出, 不需要逐点推进。")
    print()
    print("  即: 每层只需携带 2 个实体向量(端) + 1 个标尺(r[2]),")
    print("      就能同时得到该层的路径、能量、以及交给下一层的交接点。")
    print()


if __name__ == "__main__":
    part1_projections()
    part2_coeffs()
    part3_regime()
    part4_path()
    print("=" * 78)
    print("一句话")
    print("=" * 78)
    print("  交接处的路径 = '抛物线(空间) 与 弦(折线) 的交点',")
    print("  其参数 s± 是 r[2] 的初等函数 —— 这就是中间过程可以被完全跳过的原因。")
