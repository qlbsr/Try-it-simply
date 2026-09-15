# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
复现并扩展 gaji.py (概率条件目录): 分角度区间采集, 验证用户规律
  "对于不同角度区间(16,30,45,74), 概率相关性几乎都是偏差值等于角度差"

gaji.py 原设置 (run_single_trial):
  s1  = batch_probability(F1, F2, points, a)       # 理论taus提取焦点对的概率
  s2  = batch_probability(F11, F22, points, a)      # (0,1)提取焦点对的概率
  s3  = prob_total1                                 # 理论taus 格点总概率
  s4  = prob_total2                                 # (0,1) 格点总概率
  s13 = batch_probability(c*v3pca, -c*v3pca, ...)   # PCA 方向焦点概率 (实际)
  angle = ∠(v3=(F1-F2), v31=(F11-F22))             # 理论vs(0,1)焦点方向夹角

本脚本: 向量化 (fast_probs 替代纯Python格点循环), 采集多区间样本,
  输出 概率值(×100) 与 角度, 供总结条件式。
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


def run_trial_fast(seed, n_points=200):
    """gaji.run_single_trial 的向量化等价"""
    rng = np.random.default_rng(seed)
    u = rng.normal(0, 1, (n_points, 3))
    r = rng.random(n_points) ** (1 / 3)
    points = (u / np.linalg.norm(u, axis=1, keepdims=True)) * r[:, None]

    rp = np.mean(np.linalg.norm(points, axis=1))
    d2 = math.asin(math.cos(30 * math.pi / 180) / math.pi)
    a = rp * (1 + math.sin(d2))
    e3 = math.cos(d2) * 2 / (1 + math.sin(d2))
    c = rp * math.cos(d2) * e3

    _, r45, r30, _ = n2.yzqx(points, rp)
    t1, t2 = n2.compute_taus()
    t11 = t22 = complex(0, 1)

    pt1, p1, p2, sigm = dd.fast_probs(r30, r45, t1, t2, a)
    pt2, p1b, p2b, sigm2 = dd.fast_probs(r30, r45, t11, t22, a)

    F1, F2 = m.extract_foci(points, p1, sigm[1], p2, sigm[2])
    F11, F22 = m.extract_foci(points, p1b, sigm2[1], p2b, sigm2[2])

    v3pca = m.pca(points)[0]

    s1 = batch_probability(points, F1, F2, a)
    s2 = batch_probability(points, F11, F22, a)
    s3 = pt1
    s4 = pt2
    s13 = batch_probability(points, c * v3pca, -c * v3pca, a)

    v3 = (F1 - F2) / np.linalg.norm(F1 - F2)
    v31 = (F11 - F22) / np.linalg.norm(F11 - F22)
    ang = angle(v3, v31)
    return ang, s1[0], s2[0], s3[0], s4[0], s13[0]


def main():
    n_trials = 3000
    # 角度区间 [低,高): 16,30,45,74 及其对称区
    bands = [(0, 16), (16, 30), (30, 45), (45, 74), (74, 90),
             (90, 106), (106, 135), (135, 150), (150, 164), (164, 180)]
    collected = {b: [] for b in bands}
    print(f"采集 {n_trials} 次试验, 分角度区间统计", flush=True)
    for seed in range(n_trials):
        ang, p1, p2, p3, p4, actual = run_trial_fast(seed)
        for lo, hi in bands:
            if lo < ang < hi:
                collected[(lo, hi)].append((ang, p1, p2, p3, p4, actual))
                break
        if seed % 500 == 0:
            print(f"  {seed}/{n_trials}", flush=True)

    print()
    print("=" * 100)
    print("角度区间 | 样本数 | 角度均值 | p1×100 | p2×100 | p3×100 | p4×100 | actual×100")
    print("         |       |          | (理论焦点对) (0,1焦点对) (理论格点) (0,1格点) (PCA方向)")
    print("=" * 100)
    for (lo, hi), rows in collected.items():
        if not rows:
            continue
        arr = np.array(rows)
        a_ = arr[:, 0].mean()
        means = [arr[:, k].mean() * 100 for k in range(1, 6)]
        print(f"  ({lo:3d},{hi:3d}) | {len(rows):5d} | {a_:7.1f}° | "
              f"{means[0]:7.2f} | {means[1]:7.2f} | {means[2]:7.2f} | {means[3]:7.2f} | {means[4]:7.2f}")
        # 区间内 p1-p2 与角度差的相关性
        if len(rows) > 5:
            angs = arr[:, 0]
            d12 = np.abs(arr[:, 1] - arr[:, 2]) * 100
            d34 = np.abs(arr[:, 3] - arr[:, 4]) * 100
            # 角度到区间中心的距离
            mid = (lo + hi) / 2
            dev = np.abs(angs - mid)
            c12 = np.corrcoef(d12, dev)[0, 1] if dev.std() > 0 else float('nan')
            c34 = np.corrcoef(d34, dev)[0, 1] if dev.std() > 0 else float('nan')
            print(f"           corr(|p1-p2|×100, |angle-mid|)={c12:+.3f}  "
                  f"corr(|p3-p4|×100, |angle-mid|)={c34:+.3f}")
    print("DONE")


if __name__ == "__main__":
    main()
