# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""验证 nsjy5 改进: 探索=冷启动内层8步+extract_foci重拟合(便宜)+30轮+早停, 微调=NelderMead兜底"""
import math
import time

import numpy as np
import scipy.optimize as opt

import data_driven_axis as dd
import n2sjy2 as n2


def angle(u, v):
    return math.degrees(math.acos(np.clip(np.dot(u, v), -1, 1)))


def setup(pts):
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
    return r30, r45, a, v, d0


def angles_at(x, pts, r30, r45, a, v, d0):
    t1 = complex(x[0], x[1])
    t2 = complex(x[2], x[3])
    pt, p1, p2, sig = dd.fast_probs(r30, r45, t1, t2, a)
    F1, F2 = n2.extract_foci(pts, p1, sig[1], p2, sig[2])
    d = np.array(F1, float) - np.array(F2, float)
    d = d / np.linalg.norm(d)
    return (math.degrees(math.acos(np.clip(np.dot(d, v), -1, 1))),
            math.degrees(math.acos(np.clip(np.dot(d, d0), -1, 1))))


def solve_ebf_fast(pts, outer=30, inner=8, tol=10.0):
    r30, r45, a, v, d0 = setup(pts)
    d = d0.copy()
    best = {"delta": 1e9, "x": None}
    early = -1
    for k in range(outer):
        t1n, t2n, F1, F2 = dd.inner_refine(complex(0, 1), complex(0, 1),
                                           pts, r30, r45, a, d, inner)
        pt, p1, p2, sig = dd.fast_probs(r30, r45, t1n, t2n, a)
        F1f, F2f = n2.extract_foci(pts, p1, sig[1], p2, sig[2])
        d_new = np.array(F1f, float) - np.array(F2f, float)
        d_new = d_new / np.linalg.norm(d_new)
        ad, ad1 = angles_at([t1n.real, t1n.imag, t2n.real, t2n.imag], pts, r30, r45, a, v, d0)
        delta = abs(ad - ad1)
        if delta < best["delta"]:
            best = {"delta": delta, "x": [t1n.real, t1n.imag, t2n.real, t2n.imag]}
        if delta <= tol and early < 0:
            early = k
        d = d + 0.9 * (d_new - d)
        d = d / np.linalg.norm(d)
        if early >= 0:
            break
    # 微调
    ft = 0
    if best["delta"] > tol:
        evals = [0]

        def cost(x):
            evals[0] += 1
            ad, ad1 = angles_at(x, pts, r30, r45, a, v, d0)
            return abs(ad - ad1)

        def cb(xk):
            return cost(xk) <= tol

        x0 = np.array([max(-0.5, min(0.5, best["x"][0])),
                       max(0.5, min(3.0, best["x"][1])),
                       max(-0.5, min(0.5, best["x"][2])),
                       max(0.5, min(3.0, best["x"][3]))])
        bounds = [(-0.5, 0.5), (0.5, 3.0), (-0.5, 0.5), (0.5, 3.0)]
        res = opt.minimize(cost, x0, method="Nelder-Mead", bounds=bounds, callback=cb,
                           options={"maxiter": 600, "maxfev": 1200})
        ft = evals[0]
        ad, ad1 = angles_at(res.x, pts, r30, r45, a, v, d0)
        best = {"delta": abs(ad - ad1), "x": list(res.x)}
    return best, early, ft


def main():
    datasets = {"cube0": dd.make_cube(10), "cube1": dd.make_cube(11),
                "ball0": dd.make_ball(0), "ball1": dd.make_ball(1),
                "ellip0": dd.make_ellip(0), "ellip1": dd.make_ellip(1)}
    print("=" * 96)
    print("nsjy5 改进版 EBF: 探索(冷启动8步+extract_foci+30轮+早停) → 微调(NelderMead兜底)")
    print("=" * 96)
    n_ok = 0
    for name, pts in datasets.items():
        t0 = time.time()
        best, early, ft = solve_ebf_fast(pts)
        ok = "✓" if best["delta"] <= 10 else "✗"
        if best["delta"] <= 10:
            n_ok += 1
        t1f, t2f = complex(best["x"][0], best["x"][1]), complex(best["x"][2], best["x"][3])
        print(f"{name:8s} 探索best|Δ|={best['delta']:6.2f}° ({ok})  早停@轮={early}  "
              f"微调={ft:4d}求值  时间={time.time()-t0:5.1f}s  t1f=({t1f.real:+.4f},{t1f.imag:.4f}) "
              f"t2f=({t2f.real:+.4f},{t2f.imag:.4f})", flush=True)
    print(f"达标率: {n_ok}/{len(datasets)}")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
