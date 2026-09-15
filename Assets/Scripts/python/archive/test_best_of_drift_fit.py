# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""用用户的重拟合方式 (fit_foci_by_probability) 验证'取最靠近' (min|Δ|≤10)"""
import math
import time

import numpy as np

import data_driven_axis as dd
import n2sjy2 as n2


def angle(u, v):
    return math.degrees(math.acos(np.clip(np.dot(u, v), -1, 1)))


def drift_fit(pts, max_outer=120, n_inner=8, omega=0.9):
    rp = n2.compute_rp(pts)
    d2 = math.asin(math.cos(math.radians(30)) / math.pi)
    a = rp * (1 + math.sin(d2))
    _, r45, r30, _ = n2.yzqx(pts, rp)
    v = n2.pca_axis(pts)
    t1t, t2t = n2.compute_taus()
    pt0, p10, p20, sig0 = dd.fast_probs(r30, r45, t1t, t2t, a)
    F10, F20 = n2.extract_foci(pts, p10, sig0[1], p20, sig0[2])
    d0 = np.array(F10, float) - np.array(F20, float)
    d0 = d0 / np.linalg.norm(d0)
    t1 = t2 = complex(0, 1)
    d = d0.copy()
    best = {"d": 1e9, "it": -1}
    first = -1
    for k in range(max_outer):
        t1n, t2n, F1, F2 = dd.inner_refine(t1, t2, pts, r30, r45, a, d, n_inner)
        pt, p1, p2, sig = dd.fast_probs(r30, r45, t1n, t2n, a)
        F1f, F2f = n2.fit_foci_by_probability(pts, pt, F1, F2, a, max_iter=150)
        d_new = np.array(F1f, float) - np.array(F2f, float)
        d_new = d_new / np.linalg.norm(d_new)
        delta = abs(angle(d_new, v) - angle(d_new, d0))
        if delta < best["d"]:
            best = {"d": delta, "it": k}
        if delta <= 10 and first < 0:
            first = k
        chg = angle(d_new, d)
        d = d + omega * (d_new - d)
        d = d / np.linalg.norm(d)
        t1, t2 = t1n, t2n
        if chg < 0.5:
            break
    return best, first


def main():
    datasets = {"cube0": dd.make_cube(10), "ball0": dd.make_ball(0),
                "ellip0": dd.make_ellip(0), "cube1": dd.make_cube(11),
                "ball1": dd.make_ball(1), "ellip1": dd.make_ellip(1)}
    print("fit_foci_by_probability 重拟合 (用户方式): 取最靠近 min|Δ|", flush=True)
    n_ok = 0
    for name, pts in datasets.items():
        t0 = time.time()
        best, first = drift_fit(pts)
        ok = "✓" if best["d"] <= 10 else "✗"
        if best["d"] <= 10:
            n_ok += 1
        print(f"{name}: min|Δ|={best['d']:6.2f}° ({ok}) @外环={best['it']:3d}  "
              f"首达≤10@外环={first}  [{time.time()-t0:.0f}s]", flush=True)
    print(f"达标率: {n_ok}/{len(datasets)}")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
