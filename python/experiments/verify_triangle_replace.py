# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
最终验证: 三角形方法能否替代 n2sjy2.cs 的整个终止条件 (完全脱离 PCA)?
  终止条件: |angleDeg - angleDeg1| ≤ 16, angleDeg=∠(d,v)[PCA], angleDeg1=∠(d,v1)
  方案: 三角形定位 v5(当前方向) → 得到 d
        还需要 v(主轴) → 若 v 也能从三角形重建(用 s0 = P(v·c)? 但 v 未知!)
        替代: 用 v1, v3 (无PCA参考) 替代 v 的角色 → 条件改为
              |∠(d,v1) - ∠(d,v3)| ≤ 16  ?
  测试: ① 定位 d (已验证 0.05°)
        ② 验证 |∠(d,v1)-∠(d,v3)| 与 |∠(d,v)-∠(d,v1)| 的相关性 (能否替代)
        ③ 检查 v1, v3 是否接近 v (若接近则直接替代)
"""
import json
import math

import numpy as np

import data_driven_axis as dd
import n2sjy2 as n2
import nsjy_algorithms as m

PYJSON = r"C:\Users\23128\My project (2)\Assets\Resources\pyjson.json"


def angle(u, v):
    return math.degrees(math.acos(np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1)))


def batch_probability(points, F1, F2, a):
    P = np.asarray(points, float)
    d1 = np.linalg.norm(P - np.asarray(F1, float), axis=1)
    d2 = np.linalg.norm(P - np.asarray(F2, float), axis=1)
    delta = d1 + d2 - 2.0 * a
    return np.exp(-np.abs(delta) / (2.0 * a))


def build_inv_table(p_i, c, a, n=7201):
    r0 = np.linalg.norm(p_i)
    th = np.linspace(0, 180, n)
    s = np.zeros(n)
    for i, t in enumerate(th):
        rad = math.radians(t)
        d1 = math.sqrt(r0 ** 2 + c ** 2 - 2 * r0 * c * math.cos(rad))
        d2 = math.sqrt(r0 ** 2 + c ** 2 + 2 * r0 * c * math.cos(rad))
        s[i] = math.exp(-abs(d1 + d2 - 2 * a) / (2 * a))
    return th, s


def invert_candidates(s_val, th_grid, s_grid, tol=2e-3):
    near = np.where(np.abs(s_grid - s_val) < tol)[0]
    if len(near) == 0:
        i = np.argmin(np.abs(s_grid - s_val))
        near = [i]
    ths = th_grid[near]
    cands = []
    for base in (0, 90):
        mm = ths[(ths >= base) & (ths < base + 90)]
        if len(mm):
            cands.append(float(np.median(mm)))
    return cands


def cone_intersect(p0, th0_cands, p1, th1_cands):
    a0 = np.asarray(p0, float) / np.linalg.norm(p0)
    a1 = np.asarray(p1, float) / np.linalg.norm(p1)
    cos_a = np.dot(a0, a1)
    sin_a = math.sqrt(max(0.0, 1 - cos_a ** 2))
    sols = []
    for t0 in th0_cands:
        for t1 in th1_cands:
            c0 = math.cos(math.radians(t0))
            c1 = math.cos(math.radians(t1))
            if abs(sin_a) < 1e-9:
                continue
            denom = 1 - cos_a ** 2
            alpha = (c0 - c1 * cos_a) / denom
            beta = (c1 - c0 * cos_a) / denom
            gamma2 = max(0.0, 1 - alpha ** 2 - beta ** 2 - 2 * alpha * beta * cos_a)
            gamma = math.sqrt(gamma2)
            n = np.cross(a0, a1)
            n = n / np.linalg.norm(n)
            for sgn in (1.0, -1.0):
                v = alpha * a0 + beta * a1 + sgn * gamma * n
                v = v / np.linalg.norm(v)
                sols.append(v)
    return sols


def localize(P, s5_arr, idx_pairs, c, a):
    """用多点概率定位方向, 返回 (best, err)"""
    tables = {}
    for idx in set(i for pr in idx_pairs for i in pr):
        pi = P[idx]
        th, sg = build_inv_table(pi, c, a)
        tables[idx] = (th, sg, invert_candidates(s5_arr[idx], th, sg))
    all_sols = []
    for (i0, i1) in idx_pairs:
        all_sols.extend(cone_intersect(P[i0], tables[i0][2], P[i1], tables[i1][2]))
    # 用第三点消歧: 遍历第三点的所有候选 θ (双解都要试)
    extra = [i for i in range(6) if i not in (idx_pairs[0][0], idx_pairs[0][1])][0]
    th_e, sg_e = build_inv_table(P[extra], c, a)
    c_e = invert_candidates(s5_arr[extra], th_e, sg_e)
    cos_cands = [math.cos(math.radians(cc)) for cc in c_e]
    best = None
    for s in all_sols:
        dot_e = np.dot(s, P[extra] / np.linalg.norm(P[extra]))
        if any(abs(dot_e - ce) < 0.02 for ce in cos_cands):
            best = s
            break
    return best


def main():
    global c
    with open(PYJSON, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    pts = [np.array([p["x"], p["y"], p["z"]], float) for p in raw]
    P = np.array(pts, float)
    rp = n2.compute_rp(pts)
    d2 = math.asin(math.cos(math.radians(30)) / math.pi)
    a = rp * (1 + math.sin(d2))
    e3 = math.cos(d2) * 2 / (1 + math.sin(d2))
    c = rp * math.cos(d2) * e3
    _, r45, r30, _ = n2.yzqx(pts, rp)
    t1t, t2t = n2.compute_taus()
    t_ = complex(0, 1)
    pt, p1, p2, sig = dd.fast_probs(r30, r45, t1t, t2t, a)
    F1, F2 = m.extract_foci(pts, p1, sig[1], p2, sig[2])
    pt01, p101, p201, sig01 = dd.fast_probs(r30, r45, t_, t_, a)
    F01, F02 = m.extract_foci(pts, p101, sig01[1], p201, sig01[2])
    v = m.pca(pts)[1]
    v = v / np.linalg.norm(v)
    v1 = (np.array(F1, float) - np.array(F2, float))
    v1 = v1 / np.linalg.norm(v1)
    v3 = (np.array(F01, float) - np.array(F02, float))
    v3 = v3 / np.linalg.norm(v3)
    f1z, f2z = m.extract_foci(pts, pt, sig[1], pt, sig[2])
    v5_true = (np.array(f1z, float) - np.array(f2z, float))
    v5_true = v5_true / np.linalg.norm(v5_true)

    print("=" * 84)
    print("三角形替代终止条件的可行性分析")
    print("=" * 84)
    print(f"∠(v_pca, v1) = {angle(v, v1):.1f}°   ∠(v_pca, v3) = {angle(v, v3):.1f}°")
    print(f"∠(v1, v3) = {angle(v1, v3):.1f}°")
    print()

    # 定位 v5 (用不同点对组合)
    s5_arr = batch_probability(pts, v5_true * c, -v5_true * c, a)
    best = localize(P, s5_arr, [(0, 1), (0, 2)], c, a)
    if best is None:
        print("定位失败"); return
    err = angle(best, v5_true)
    print(f"三角形定位 v5: 误差 {err:.2f}°")
    print(f"  真值: ∠(d,v)={angle(v5_true, v):.1f}° ∠(d,v1)={angle(v5_true, v1):.1f}° |Δ|={abs(angle(v5_true, v)-angle(v5_true, v1)):.1f}°")
    print(f"  重建: ∠(d,v1)={angle(best, v1):.1f}° ∠(d,v3)={angle(best, v3):.1f}°")
    print()

    # 关键: 终止条件需要 angleDeg=∠(d,v) [PCA]. 三角形重建出 d 但 v 仍需 PCA!
    # 替代方案: 用 v1 或 v3 替代 v 的角色 → 新条件 |∠(d,v1)-∠(d,v3)| ≤ ?
    a_d1 = angle(best, v1)
    a_d3 = angle(best, v3)
    a_dv = angle(best, v)
    print("=" * 84)
    print("替代条件分析: 原条件 |∠(d,v) - ∠(d,v1)| ≤ 16")
    print("=" * 84)
    print(f"  原: |∠(d,v) - ∠(d,v1)| = |{a_dv:.1f} - {a_d1:.1f}| = {abs(a_dv-a_d1):.1f}°")
    print(f"  替代A: |∠(d,v1) - ∠(d,v3)| = |{a_d1:.1f} - {a_d3:.1f}| = {abs(a_d1-a_d3):.1f}°")
    print(f"  替代B: ∠(d,v1) 本身 = {a_d1:.1f}°  (v1 不依赖 PCA)")
    print()
    print("结论:")
    print("  1. 三角形可定位 d (误差~0.05°) → ∠(d,v1) 完全可得, 无 PCA ✓")
    print("  2. ∠(d,v) 需要 v=PCA 坐标 → 三角形无法凭空得到 PCA 轴")
    print("  3. 若 v1 与 v 夹角小 → ∠(d,v1)≈∠(d,v), 可近似替代 (需数据验证)")
    print("DONE")


if __name__ == "__main__":
    main()
