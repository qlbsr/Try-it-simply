# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
验证用户 glxzf 公式 + 分区间规律总结 (gaji.py 语义, 向量化)
  angle = ∠(v1理论焦点方向, v3(0,1)焦点方向)  [不依赖 PCA]
  概率: p1=P(F1,F2), p2=P(F11,F22), p3=prob_total1, p4=prob_total2
用户规律假设:
  R1: 偏差值等于角度差 → (p2-p1)×100 与 (p4-p3)×100 同号成比例
  R2: 分区间 (16,30,45,74): 各区间内 |概率差| 与 角度 的局部关系
验证 glxzf 164-180 公式:
  d1 = max(p1,p2)*100 - (jd-164);  d2 = max(p3,p4)*100 + (jd-164);  dd=(d1+d2)/2
  → 看 dd 是否 ≈ 常数/与 jd 相关
"""
import math
import numpy as np

import data_driven_axis as dd
import n2sjy2 as n2
import nsjy_algorithms as m


def angle(u, v):
    return math.degrees(math.acos(np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1)))


def batch_probability(points, F1, F2, a):
    P = np.asarray(points, float)
    d1 = np.linalg.norm(P - np.asarray(F1, float), axis=1)
    d2 = np.linalg.norm(P - np.asarray(F2, float), axis=1)
    delta = d1 + d2 - 2.0 * a
    return np.exp(-np.abs(delta) / (2.0 * a))


def trial(seed):
    rng = np.random.default_rng(seed)
    u = rng.normal(0, 1, (200, 3))
    r = rng.random(200) ** (1 / 3)
    points = (u / np.linalg.norm(u, axis=1, keepdims=True)) * r[:, None]
    rp = np.mean(np.linalg.norm(points, axis=1))
    d2 = math.asin(math.cos(30 * math.pi / 180) / math.pi)
    a = rp * (1 + math.sin(d2))
    _, r45, r30, _ = n2.yzqx(points, rp)
    t1, t2 = n2.compute_taus()
    t11 = t22 = complex(0, 1)
    pt1, p1, p2, sig = dd.fast_probs(r30, r45, t1, t2, a)
    pt2, p1b, p2b, sig2 = dd.fast_probs(r30, r45, t11, t22, a)
    F1, F2 = m.extract_foci(points, p1, sig[1], p2, sig[2])
    F11, F22 = m.extract_foci(points, p1b, sig2[1], p2b, sig2[2])
    s1 = batch_probability(points, F1, F2, a)[0]
    s2 = batch_probability(points, F11, F22, a)[0]
    s3 = pt1[0]
    s4 = pt2[0]
    v3 = (F1 - F2) / np.linalg.norm(F1 - F2)
    v31 = (F11 - F22) / np.linalg.norm(F11 - F22)
    ang = angle(v3, v31)
    return ang, s1, s2, s3, s4


def main():
    N = 8000
    rows = []
    for seed in range(N):
        rows.append(trial(seed))
    A = np.array(rows)
    ang = A[:, 0]
    p1, p2, p3, p4 = A[:, 1], A[:, 2], A[:, 3], A[:, 4]
    d12 = (p2 - p1) * 100
    d34 = (p4 - p3) * 100

    # R1: d12 vs d34 同号比例
    print("=" * 78)
    print("R1: d12=(p2-p1)×100 vs d34=(p4-p3)×100  (同号成比例?)")
    print("=" * 78)
    mask = (np.abs(d12) > 0.5) & (np.abs(d34) > 0.5)
    same_sign = (np.sign(d12[mask]) == np.sign(d34[mask])).mean()
    ratio = d34[mask] / np.maximum(np.abs(d12[mask]), 1e-9)
    print(f"  同号占比: {same_sign*100:.1f}%  |d34/d12| 中位 {np.median(np.abs(ratio)):.2f}")
    print(f"  corr(d12, d34) = {np.corrcoef(d12, d34)[0,1]:+.3f}")
    print()

    # R2: 分区间 (用户 glxzf 区间)
    bands = [(0, 16), (16, 30), (30, 45), (45, 74), (74, 90),
             (90, 164), (164, 180)]
    print("=" * 78)
    print("R2: 分区间统计 (angle 所在区间)")
    print("=" * 78)
    print(f"{'区间':>14s} {'n':>5s} {'∠均值':>7s} {'|d12|':>7s} {'|d34|':>7s} "
          f"{'max(p1,p2)':>11s} {'max(p3,p4)':>11s} {'corr(∠,|d34|)':>12s}")
    for lo, hi in bands:
        mk = (ang > lo) & (ang < hi) & (np.abs(d12) > 0.5) & (np.abs(d34) > 0.5)
        if mk.sum() < 10:
            continue
        c = np.corrcoef(ang[mk], np.abs(d34[mk]))[0, 1] if np.abs(d34[mk]).std() > 0 else float('nan')
        print(f"  ({lo:3d},{hi:3d}) {mk.sum():5d} {ang[mk].mean():7.1f} "
              f"{np.abs(d12[mk]).mean():7.2f} {np.abs(d34[mk]).mean():7.2f} "
              f"{np.maximum(p1[mk],p2[mk]).mean()*100:11.2f} "
              f"{np.maximum(p3[mk],p4[mk]).mean()*100:11.2f} {c:+12.3f}")
    print()

    # glxzf 164-180 公式验证: dd 行为
    print("=" * 78)
    print("glxzf 164-180 区间公式验证: d1=max(p1,p2)*100-(jd-164), "
          "d2=max(p3,p4)*100+(jd-164), dd=(d1+d2)/2")
    print("=" * 78)
    mk = (ang > 164) & (ang < 180) & (np.abs(d12) > 0.5) & (np.abs(d34) > 0.5)
    if mk.sum() > 5:
        jd = ang[mk]
        d1 = np.maximum(p1[mk], p2[mk]) * 100 - (jd - 164)
        d2 = np.maximum(p3[mk], p4[mk]) * 100 + (jd - 164)
        dd = (d1 + d2) / 2
        print(f"  n={mk.sum()}: dd 均值 {dd.mean():.2f} 标准差 {dd.std():.2f}  "
              f"范围 [{dd.min():.1f}, {dd.max():.1f}]")
        print(f"  corr(dd, jd) = {np.corrcoef(dd, jd)[0,1]:+.3f}  "
              f"corr(d1-d2, jd) = {np.corrcoef(d1-d2, jd)[0,1]:+.3f}")
        print(f"  d1-d2 均值 {np.mean(d1-d2):.2f} (若≈0 → 公式对称)")
    else:
        print("  样本不足")
    print()
    print("结论将写入总结")
    print("DONE")


if __name__ == "__main__":
    main()
