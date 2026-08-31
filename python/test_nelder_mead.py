# -*- coding: utf-8 -*-
"""
目标: 快速满足 if ((angleDeg+angleDeg1)/2 - min(angleDeg,angleDeg1)) <= 5  即 |Δ|<=10°
方法: 无梯度优化 Nelder-Mead / Powell, 4维 x=(Re t1,Im t1,Re t2,Im t2),
      边界=基本域箱, 回调提前停止, 统计达到条件的函数求值次数
"""
import math
import time

import numpy as np
import scipy.optimize as opt

import data_driven_axis as dd
import n2sjy2 as n2


def angles_at(x, pts, r30, r45, a, v_pca, d_init):
    t1 = complex(x[0], x[1])
    t2 = complex(x[2], x[3])
    pt, p1, p2, sig = dd.fast_probs(r30, r45, t1, t2, a)
    F1, F2 = n2.extract_foci(pts, p1, sig[1], p2, sig[2])
    d = np.array(F1, float) - np.array(F2, float)
    nrm = np.linalg.norm(d)
    if nrm < 1e-12:
        return 180.0, 180.0
    d = d / nrm
    ad = math.degrees(math.acos(np.clip(np.dot(d, v_pca), -1, 1)))
    ad1 = math.degrees(math.acos(np.clip(np.dot(d, d_init), -1, 1)))
    return ad, ad1


def setup(pts):
    rp = n2.compute_rp(pts)
    d2 = math.asin(math.cos(math.radians(30)) / math.pi)
    a = rp * (1 + math.sin(d2))
    _, r45, r30, _ = n2.yzqx(pts, rp)
    v_pca = n2.pca_axis(pts)
    t1t, t2t = n2.compute_taus()
    pt0, p10, p20, sig0 = dd.fast_probs(r30, r45, t1t, t2t, a)
    F10, F20 = n2.extract_foci(pts, p10, sig0[1], p20, sig0[2])
    d_init = np.array(F10, float) - np.array(F20, float)
    d_init = d_init / np.linalg.norm(d_init)
    return r30, r45, a, v_pca, d_init


def solve(pts, method, x0, max_evals=1000):
    r30, r45, a, v_pca, d_init = setup(pts)
    evals = [0]
    best = [1e9]

    def cost(x):
        evals[0] += 1
        ad, ad1 = angles_at(x, pts, r30, r45, a, v_pca, d_init)
        c = abs(ad - ad1)
        if c < best[0]:
            best[0] = c
        return c

    def cb(xk):
        return cost(xk) <= 10.0          # 达到条件即停

    bounds = [(-0.5, 0.5), (0.5, 3.0), (-0.5, 0.5), (0.5, 3.0)]
    opts = {'maxiter': max_evals, 'xatol': 1e-3, 'fatol': 1e-3,
            'maxfev': max_evals * 2}
    res = opt.minimize(cost, np.array(x0, float), method=method, bounds=bounds,
                       callback=cb, options=opts)
    ad_f, ad1_f = angles_at(res.x, pts, r30, r45, a, v_pca, d_init)
    return evals[0], abs(ad_f - ad1_f), ad_f, ad1_f, best[0]


def main():
    datasets = {"cube0": dd.make_cube(10), "cube1": dd.make_cube(11),
                "ball0": dd.make_ball(0), "ball1": dd.make_ball(1),
                "ellip0": dd.make_ellip(0), "ellip1": dd.make_ellip(1),
                "gauss0": dd.make_gauss(0) if hasattr(dd, "make_gauss") else None}
    if datasets["gauss0"] is None:
        del datasets["gauss0"]
    starts = {"(0,1)": [0, 1, 0, 1],
              "归一化理论": [0, 1.2793, 0, 1]}

    print("=" * 108)
    print("无梯度优化达到 |Δ|<=10° 的求值次数 (x=(Re t1,Im t1,Re t2,Im t2), 边界=基本域箱)")
    print("=" * 108)
    for method in ("Nelder-Mead", "Powell"):
        print(f"--- {method} ---")
        n_ok = 0
        for name, pts in datasets.items():
            line = f"{name:8s} "
            for sname, x0 in starts.items():
                t0 = time.time()
                ev, df, ad, ad1, best = solve(pts, method, x0)
                ok = "✓" if df <= 10 else "✗"
                if df <= 10:
                    n_ok += 1
                line += (f"{sname}: evals={ev:4d} |Δ|={df:5.1f}° ({ok}) "
                         f"best={best:5.1f}° [{time.time()-t0:4.0f}s]   ")
            print(line)
        print(f"  {method} 达标次数: {n_ok}/{len(datasets)*len(starts)}")
        print()
    print("DONE")


if __name__ == "__main__":
    main()
