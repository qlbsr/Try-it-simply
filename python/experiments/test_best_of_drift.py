# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
验证"取最靠近": n2sjy2 漂移轨迹上跟踪 min|Δ|, 是否 ≤10 (几乎总是)?
并测: 早停(首达≤10的外环序号), best 点是否可用作 nsjy4 微调热启动.
快变体: 热内层8步 + extract_foci 重拟合 + 阻尼0.9, 外层上限200
"""
import math
import time

import numpy as np

import data_driven_axis as dd
import n2sjy2 as n2


def angle(u, v):
    return math.degrees(math.acos(np.clip(np.dot(u, v), -1, 1)))


def drift_track_best(pts, max_outer=200, n_inner=8, omega=0.9, tol_deg=0.5):
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
    best = {"delta": 1e9, "it": -1, "t1": t1, "t2": t2, "d": d.copy()}
    first_le10 = -1
    hist = []
    for k in range(max_outer):
        # 内层 LM (轴=当前d, 热启动)
        t1n, t2n, F1, F2 = dd.inner_refine(t1, t2, pts, r30, r45, a, d, n_inner)
        # 自由重拟合
        pt, p1, p2, sig = dd.fast_probs(r30, r45, t1n, t2n, a)
        F1f, F2f = n2.extract_foci(pts, p1, sig[1], p2, sig[2])
        d_new = np.array(F1f, float) - np.array(F2f, float)
        d_new = d_new / np.linalg.norm(d_new)
        delta = abs(angle(d_new, v) - angle(d_new, d0))
        hist.append(delta)
        if delta < best["delta"]:
            best = {"delta": delta, "it": k, "t1": t1n, "t2": t2n, "d": d_new.copy()}
        if delta <= 10.0 and first_le10 < 0:
            first_le10 = k
        # 阻尼更新
        chg = angle(d_new, d)
        d = d + omega * (d_new - d)
        d = d / np.linalg.norm(d)
        t1, t2 = t1n, t2n
        if chg < tol_deg:
            break
    return best, first_le10, len(hist), d0, v, r30, r45, a


def main():
    datasets = {"cube0": dd.make_cube(10), "cube1": dd.make_cube(11),
                "ball0": dd.make_ball(0), "ball1": dd.make_ball(1),
                "ellip0": dd.make_ellip(0), "ellip1": dd.make_ellip(1)}
    print("=" * 108)
    print("n2sjy2 漂移取最靠近: min|Δ|, 首达≤10的外环, 轨迹长度")
    print("=" * 108)
    n_ok = 0
    for name, pts in datasets.items():
        t0 = time.time()
        best, first_le10, L, d0, v, r30, r45, a = drift_track_best(pts)
        ok = "✓" if best["delta"] <= 10 else "✗"
        if best["delta"] <= 10:
            n_ok += 1
        print(f"{name:8s} min|Δ|={best['delta']:6.2f}° ({ok})  @外环={best['it']:3d}  "
              f"首达≤10 @外环={first_le10}  轨迹长={L:3d}  "
              f"∠(best.d,v)={angle(best['d'], v):6.1f}°  [{time.time()-t0:.0f}s]")
    print(f"\n达标率 (min|Δ|≤10): {n_ok}/{len(datasets)}")
    print("DONE")


if __name__ == "__main__":
    main()
