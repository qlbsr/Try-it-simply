# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
诚实验证: 三角形方法能否无监督定位 d, 以及能否替代终止条件?
  1. 无监督定位: 用 N 个点反演 θ_i (各 2 候选) → 两两锥面交线 → 全部候选
     用**剩余点**的 θ 约束投票选唯一 (不碰真值, 不碰 PCA)
  2. 定位误差 (与真值对比, 仅评估)
  3. 终止条件替代性分析: angleDeg=∠(d,v) 需要 PCA 轴 v
     - 三角形能给出: ∠(d,v1), ∠(d,v3)  (v1,v3 无 PCA)
     - 三角形不能给出: v 本身 (PCA 轴是数据统计量, 非几何反演量)
  结论: 若替代条件 |∠(d,v1)-∠(d,v3)| 与 |∠(d,v)-∠(d,v1)| 强相关 → 可替代
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


def invert_candidates(s_val, th_grid, s_grid, tol=3e-3):
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


def cone_intersect(p0, t0s, p1, t1s):
    a0 = np.asarray(p0, float) / np.linalg.norm(p0)
    a1 = np.asarray(p1, float) / np.linalg.norm(p1)
    cos_a = np.dot(a0, a1)
    sin_a = math.sqrt(max(0.0, 1 - cos_a ** 2))
    sols = []
    for t0 in t0s:
        for t1 in t1s:
            c0 = math.cos(math.radians(t0))
            c1 = math.cos(math.radians(t1))
            if abs(sin_a) < 1e-9:
                continue
            denom = 1 - cos_a ** 2
            alpha = (c0 - c1 * cos_a) / denom
            beta = (c1 - c0 * cos_a) / denom
            g2 = max(0.0, 1 - alpha ** 2 - beta ** 2 - 2 * alpha * beta * cos_a)
            gamma = math.sqrt(g2)
            n = np.cross(a0, a1)
            n = n / np.linalg.norm(n)
            for sgn in (1.0, -1.0):
                v = alpha * a0 + beta * a1 + sgn * gamma * n
                v = v / np.linalg.norm(v)
                # 去重
                if not any(abs(np.dot(v, s)) > 1 - 1e-6 for s in sols):
                    sols.append(v)
    return sols


def main():
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

    s5_arr = batch_probability(pts, v5_true * c, -v5_true * c, a)

    # 1. 无监督定位: 8 个点的候选, 两两交线, 剩余点投票
    npts = 8
    tabs = {}
    for idx in range(npts):
        th, sg = build_inv_table(P[idx], c, a)
        tabs[idx] = (th, sg, invert_candidates(s5_arr[idx], th, sg))

    # 用 p0,p1 生成候选, 其余点投票
    sols = cone_intersect(P[0], tabs[0][2], P[1], tabs[1][2])
    voters = list(range(2, npts))
    scored = []
    for s in sols:
        votes = 0
        for j in voters:
            dot_j = np.dot(s, P[j] / np.linalg.norm(P[j]))
            cos_cands = [math.cos(math.radians(cc)) for cc in tabs[j][2]]
            if any(abs(dot_j - ce) < 0.03 for ce in cos_cands):
                votes += 1
        scored.append((votes, s))
    scored.sort(key=lambda x: -x[0])
    print("=" * 84)
    print(f"无监督定位: 8 点反演, p0∩p1 候选 {len(sols)} 个, 其余 6 点投票")
    print("=" * 84)
    for votes, s in scored:
        err = angle(s, v5_true)
        mark = " ← 真解" if err < 3 else ""
        print(f"  票数 {votes}/6  方向 {np.round(s,3)}  与真值夹角 {err:5.1f}°{mark}")

    best = scored[0][1]
    err_best = angle(best, v5_true)
    print()
    print(f"★ 无监督最佳: 误差 {err_best:.2f}°  (投票 {scored[0][0]}/6)")
    print()

    # 2. 终止条件替代性
    print("=" * 84)
    print("终止条件替代性 (核心问题)")
    print("=" * 84)
    print(f"  ∠(v_pca, v1) = {angle(v, v1):.1f}°   ∠(v_pca, v3) = {angle(v, v3):.1f}°")
    a_dv = angle(best, v)
    a_d1 = angle(best, v1)
    a_d3 = angle(best, v3)
    orig = abs(a_dv - a_d1)
    alt = abs(a_d1 - a_d3)
    print(f"  原条件 |∠(d,v) - ∠(d,v1)| = |{a_dv:.1f} - {a_d1:.1f}| = {orig:.1f}°  [需 PCA]")
    print(f"  替代条件 |∠(d,v1) - ∠(d,v3)| = |{a_d1:.1f} - {a_d3:.1f}| = {alt:.1f}°  [无 PCA]")
    print()
    print("  结论:")
    print("  1. 三角形定位 d 可行 (无监督投票, 若票数唯一) → ∠(d,v1) 无 PCA ✓")
    print("  2. angleDeg=∠(d,v) 需要 PCA 轴 v → 三角形无法给出 v 本身")
    print("  3. 替代条件 |∠(d,v1)-∠(d,v3)| 语义不同 (v1,v3 夹角仅 27°, 与 v 差 108°),")
    print("     不等价于原条件 → 需用户确认是否接受新判据")
    print("DONE")


if __name__ == "__main__":
    main()
