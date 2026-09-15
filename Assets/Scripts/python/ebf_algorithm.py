# -*- coding: utf-8 -*-
"""
探索→最佳点→微调 (EBF) 三件套:
  阶段1 探索 (n2sjy2 冷启动漂移, 无PCA): 每轮算 |Δ|, 跟踪 best(t1,t2), |Δ|≤10 早停
  阶段2 最佳点: min|Δ| 对应的 (t1,t2)
  阶段3 微调 (nsjy4): 从最佳点热启动 Nelder-Mead, 目标 |Δ|≤10, 上限400次求值
报告: 探索轮数/探索best|Δ|/微调求值/最终|Δ|/总时间/最终(t1,t2)
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
    nrm = np.linalg.norm(d)
    if nrm < 1e-12:
        return 180.0, 180.0
    d = d / nrm
    return (math.degrees(math.acos(np.clip(np.dot(d, v), -1, 1))),
            math.degrees(math.acos(np.clip(np.dot(d, d0), -1, 1))))


def explore(pts, r30, r45, a, v, d0, outer=80, inner=15, omega=0.9):
    """阶段1: 冷启动 n2sjy2 漂移 (无PCA), 跟踪 best (t1,t2) by |Δ|, |Δ|≤10 早停"""
    t1 = t2 = complex(0, 1)
    d = d0.copy()
    best = {"delta": 1e9, "x": None}
    early = -1
    used = 0
    for k in range(outer):
        t1n, t2n, F1, F2 = dd.inner_refine(complex(0, 1), complex(0, 1),
                                           pts, r30, r45, a, d, inner)   # 冷启动
        pt, p1, p2, sig = dd.fast_probs(r30, r45, t1n, t2n, a)
        F1f, F2f = n2.fit_foci_by_probability(pts, pt, F1, F2, a, max_iter=100)
        d_new = np.array(F1f, float) - np.array(F2f, float)
        d_new = d_new / np.linalg.norm(d_new)
        delta = abs(angle(d_new, v) - angle(d_new, d0))
        if delta < best["delta"]:
            best = {"delta": delta,
                    "x": [t1n.real, t1n.imag, t2n.real, t2n.imag]}
        if delta <= 10 and early < 0:
            early = k
        d = d + omega * (d_new - d)
        d = d / np.linalg.norm(d)
        used = k + 1
        if early >= 0:
            break
    return best, early, used


def fine_tune(pts, r30, r45, a, v, d0, x0, max_evals=400):
    """阶段3: nsjy4 从最佳点热启动, 目标 |Δ|≤10"""
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
    ad, ad1 = angles_at(res.x, pts, r30, r45, a, v, d0)
    return res.x, evals[0], abs(ad - ad1), ad, ad1


def main():
    datasets = {"cube0": dd.make_cube(10), "cube1": dd.make_cube(11),
                "ball0": dd.make_ball(0), "ball1": dd.make_ball(1),
                "ellip0": dd.make_ellip(0), "ellip1": dd.make_ellip(1)}
    print("=" * 118)
    print("EBF 探索→最佳点→微调: 探索=冷启动n2sjy2(无PCA,≤80外环,早停), 微调=nsjy4(≤400求值)")
    print("=" * 118)
    print(f"{'name':8s} {'探索轮':>5s} {'探索best|Δ|':>10s} {'微调求值':>8s} {'最终|Δ|':>7s} "
          f"{'∠(d,v)':>7s} | {'t1f':>14s} {'t2f':>14s} {'时间':>6s}")
    for name, pts in datasets.items():
        r30, r45, a, v, d0 = setup(pts)
        t0 = time.time()
        best, early, used = explore(pts, r30, r45, a, v, d0)
        if best["delta"] <= 10:
            xf, ev, delf, ad, ad1 = best["x"], 0, best["delta"], 0, 0
            # 方向
            ad, ad1 = angles_at(xf, pts, r30, r45, a, v, d0)
            t1f, t2f = complex(xf[0], xf[1]), complex(xf[2], xf[3])
        else:
            xf, ev, delf, ad, ad1 = fine_tune(pts, r30, r45, a, v, d0, best["x"])
            t1f, t2f = complex(xf[0], xf[1]), complex(xf[2], xf[3])
        ok = "✓" if delf <= 10 else "✗"
        print(f"{name:8s} {used:5d} {best['delta']:10.2f} {ev:8d} {delf:7.2f}{ok} "
              f"{ad:7.1f} | ({t1f.real:+.4f},{t1f.imag:.4f}) ({t2f.real:+.4f},{t2f.imag:.4f}) "
              f"[{time.time()-t0:.0f}s]")
    print("DONE")


if __name__ == "__main__":
    main()
