# -*- coding: utf-8 -*-
"""
真正要求验证: angle(d, v) ≤ 10° (方向离PCA 10°以内, PCA=目标)
  NM 直接最小化 angle(d,v) (不是 |Δ|)
  对比: NM最小化|Δ| (等分线) 的结果
  → 回答: 方向能否对准v? NM(等分线) vs NM(对准v) 的关系
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


def direction_at(x, pts, r30, r45, a):
    t1 = complex(x[0], x[1])
    t2 = complex(x[2], x[3])
    pt, p1, p2, sig = dd.fast_probs(r30, r45, t1, t2, a)
    F1, F2 = n2.extract_foci(pts, p1, sig[1], p2, sig[2])
    d = np.array(F1, float) - np.array(F2, float)
    return d / np.linalg.norm(d)


def nm_minimize(pts, objective, x0, max_evals=600, stop=10.0):
    r30, r45, a, v, d0 = setup(pts)
    evals = [0]

    def cost(x):
        evals[0] += 1
        return objective(x, pts, r30, r45, a, v, d0)

    def cb(xk):
        return cost(xk) <= stop

    x0c = np.array([max(-0.5, min(0.5, x0[0])), max(0.5, min(3.0, x0[1])),
                    max(-0.5, min(0.5, x0[2])), max(0.5, min(3.0, x0[3]))])
    bounds = [(-0.5, 0.5), (0.5, 3.0), (-0.5, 0.5), (0.5, 3.0)]
    res = opt.minimize(cost, x0c, method="Nelder-Mead", bounds=bounds, callback=cb,
                       options={"maxiter": max_evals, "maxfev": max_evals * 2})
    return res.x, evals[0]


def obj_angle_v(x, pts, r30, r45, a, v, d0):
    return angle(direction_at(x, pts, r30, r45, a), v)


def obj_delta(x, pts, r30, r45, a, v, d0):
    d = direction_at(x, pts, r30, r45, a)
    return abs(angle(d, v) - angle(d, d0))


def main():
    datasets = {"cube0": dd.make_cube(10), "cube1": dd.make_cube(11),
                "ball0": dd.make_ball(0), "ball1": dd.make_ball(1),
                "ellip0": dd.make_ellip(0), "ellip1": dd.make_ellip(1)}
    rng = np.random.default_rng(1000 + 5)
    datasets["gauss5"] = [rng.standard_normal(3) * (0.5 + 2 * rng.random()) for _ in range(200)]

    print("=" * 104)
    print("真正要求: angle(d,v) ≤ 10° (PCA=目标)   | NM直接最小化 angle(d,v) vs 最小化|Δ|")
    print("=" * 104)
    print(f"{'name':8s} {'∠(d0,v)':>8s} | {'NM(angle_v) 最小':>14s} {'≤10':>4s} "
          f"{'求值':>5s} | {'NM(|Δ|) 后∠(d,v)':>15s} {'等分线∠(d,v)':>11s}")
    for name, pts in datasets.items():
        t0 = time.time()
        r30, r45, a, v, d0 = setup(pts)
        t1t, t2t = n2.compute_taus()
        x0 = [t1t.real, t1t.imag, t2t.real, t2t.imag]
        # NM 直接最小化 angle(d,v)
        xf, ev = nm_minimize(pts, obj_angle_v, x0)
        min_ang = angle(direction_at(xf, pts, r30, r45, a), v)
        ok = "✓" if min_ang <= 10 else "✗"
        # NM 最小化 |Δ| (等分线) 后 的 ∠(d,v)
        xd, _ = nm_minimize(pts, obj_delta, x0)
        ang_bis = angle(direction_at(xd, pts, r30, r45, a), v)
        bis = v + d0
        bis = bis / np.linalg.norm(bis)
        print(f"{name:8s} {angle(d0, v):8.1f} | {min_ang:14.2f} {ok:>4s} {ev:5d} | "
              f"{ang_bis:15.1f} {angle(bis, v):11.1f}  [{time.time()-t0:.0f}s]", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
