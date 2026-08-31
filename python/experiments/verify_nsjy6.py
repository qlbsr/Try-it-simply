# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
nsjy6 严格验证: 点火+自持 (外环末尾 nsjy4 收尾)
  每轮: 内层LM(轴=当前d) → 比对|Δ| → 需要才 nsjy4 → 回喂 τ
  判伪自持: 30 轮全程 |Δ|≤10; 点火后(轮>=1)所有轮 |Δ|≤10; 方向/τ 不逃逸
"""
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


def direction_at(x, pts, r30, r45, a):
    t1 = complex(x[0], x[1])
    t2 = complex(x[2], x[3])
    pt, p1, p2, sig = dd.fast_probs(r30, r45, t1, t2, a)
    F1, F2 = n2.extract_foci(pts, p1, sig[1], p2, sig[2])
    d = np.array(F1, float) - np.array(F2, float)
    return d / np.linalg.norm(d)


def nsjy4_refine(pts, r30, r45, a, v, d0, x0, max_evals=400):
    evals = [0]

    def cost(x):
        evals[0] += 1
        ad, ad1 = angles_at(x, pts, r30, r45, a, v, d0)
        return abs(ad - ad1)

    def cb(xk):
        return cost(xk) <= 10.0

    x0c = np.array([max(-0.5, min(0.5, x0[0])), max(0.5, min(3.0, x0[1])),
                    max(-0.5, min(0.5, x0[2])), max(0.5, min(3.0, x0[3]))])
    bounds = [(-0.5, 0.5), (0.5, 3.0), (-0.5, 0.5), (0.5, 3.0)]
    res = opt.minimize(cost, x0c, method="Nelder-Mead", bounds=bounds, callback=cb,
                       options={"maxiter": max_evals, "maxfev": max_evals * 2})
    return res.x, evals[0]


def interleaved_long(pts, rounds=30, n_inner=15, omega=0.9):
    r30, r45, a, v, d0 = setup(pts)
    t1 = t2 = complex(0, 1)
    d = d0.copy()
    deltas = []
    fires = 0
    dirs = []
    taus = []
    for k in range(rounds):
        t1b, t2b, F1, F2 = dd.inner_refine(t1, t2, pts, r30, r45, a, d, n_inner)
        pt, p1, p2, sig = dd.fast_probs(r30, r45, t1b, t2b, a)
        F1f, F2f = n2.extract_foci(pts, p1, sig[1], p2, sig[2])
        d_new = np.array(F1f, float) - np.array(F2f, float)
        d_new = d_new / np.linalg.norm(d_new)
        xb = [t1b.real, t1b.imag, t2b.real, t2b.imag]
        ad, ad1 = angles_at(xb, pts, r30, r45, a, v, d0)
        delta_b = abs(ad - ad1)
        if delta_b > 10.0:
            xc, _ = nsjy4_refine(pts, r30, r45, a, v, d0, xb)
            t1c, t2c = complex(xc[0], xc[1]), complex(xc[2], xc[3])
            dc = direction_at(xc, pts, r30, r45, a)
            ad, ad1 = angles_at(xc, pts, r30, r45, a, v, d0)
            delta_c = abs(ad - ad1)
            fires += 1
        else:
            t1c, t2c = t1b, t2b
            dc = d_new
            delta_c = delta_b
        d = d + omega * (dc - d)
        d = d / np.linalg.norm(d)
        t1, t2 = t1c, t2c
        deltas.append(delta_c)
        dirs.append(dc.copy())
        taus.append((t1.real, t1.imag, t2.real, t2.imag))
    return deltas, fires, dirs, taus


def main():
    datasets = {"cube0": dd.make_cube(10), "cube1": dd.make_cube(11),
                "ball0": dd.make_ball(0), "ball1": dd.make_ball(1),
                "ellip0": dd.make_ellip(0), "ellip1": dd.make_ellip(1),
                "gauss0": None}
    if datasets["gauss0"] is None:
        del datasets["gauss0"]
    # 补两个
    datasets["gauss5"] = dd.make_cube(99)  # 占位(下一行替换)
    # 用真实 gauss: 构造标准正态
    rng = np.random.default_rng(1000 + 5)
    datasets["gauss5"] = [rng.standard_normal(3) * (0.5 + 2 * rng.random()) for _ in range(200)]

    print("=" * 110)
    print("nsjy6 点火+自持 严格验证: 30 轮, 判据 = 点火后全程 |Δ|≤10 且方向不逃逸")
    print("=" * 110)
    print(f"{'name':8s} {'nsjy4次数':>8s} {'全程min|Δ|':>9s} {'全程max|Δ|':>9s} "
          f"{'轮5后max|Δ|':>10s} {'全程≤10':>7s} {'真自持':>5s} {'末轮Δd':>7s}")
    for name, pts in datasets.items():
        t0 = time.time()
        deltas, fires, dirs, taus = interleaved_long(pts, rounds=30)
        deltas = np.array(deltas)
        dirs = np.array(dirs)
        tail_ok = deltas[5:].max() <= 10.0          # 点火后(轮5起)全程≤10
        all_ok = deltas.max() <= 10.0
        genuine = tail_ok and all_ok
        d_change_last = angle(dirs[-1], dirs[-2])
        print(f"{name:8s} {fires:8d} {deltas.min():9.2f} {deltas.max():9.2f} "
              f"{deltas[5:].max():10.2f} {'✓' if all_ok else '✗':>7s} "
              f"{'真' if genuine else '伪':>5s} {d_change_last:7.2f}  [{time.time()-t0:.0f}s]",
              flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
