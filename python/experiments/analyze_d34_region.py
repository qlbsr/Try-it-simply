# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
d34=(p4-p3)*100 作为角度区间指示器 (无 PCA)
结论来源: analyze_glxzf R2 — |d34| 随 angle 单调: 4.31@10° → 12.30@170°
"""
import math
import numpy as np

import data_driven_axis as dd
import n2sjy2 as n2
import nsjy_algorithms as m


def angle(u, v):
    return math.degrees(math.acos(np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1)))


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
    s3 = pt1[0]
    s4 = pt2[0]
    v3 = (F1 - F2) / np.linalg.norm(F1 - F2)
    v31 = (F11 - F22) / np.linalg.norm(F11 - F22)
    ang = angle(v3, v31)
    return ang, s3, s4


def main():
    rows = []
    for seed in range(2500):
        ang, s3, s4 = trial(seed)
        rows.append((ang, abs(s4 - s3) * 100))     # (angle, d34)
    A = np.array(rows)
    print("|d34|=(p4-p3)*100 分箱 → angle 分布 (判断角度区间, 无 PCA)")
    print("  d34箱    n  angle均值   angle范围")
    for lo in range(0, 26, 2):
        mk = (A[:, 1] >= lo) & (A[:, 1] < lo + 2)
        if mk.sum() >= 15:
            print("  [%3.0f,%3.0f) %5d %9.1f  [%5.1f, %5.1f]" % (
                lo, lo + 2, mk.sum(), A[mk, 0].mean(), A[mk, 0].min(), A[mk, 0].max()))
    print()
    print("若 |d34| 窄区间锁定 angle → 用 |d34| 判断角度区间, 替代 PCA 角度")
    print("DONE")


if __name__ == "__main__":
    main()
