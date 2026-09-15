# -*- coding: utf-8 -*-
"""
融合: n2sjy2 独立自学习(无PCA) + nsjy4 快速确认(PCA作参考, τ热启动)
  阶段1: n2sjy2 数据驱动漂移 → d_n2, τ_n2   (独立性/自学习通道, 无PCA)
  阶段2: nsjy4 Nelder-Mead 从 τ_n2 热启动, 目标 |angle(d,v)-angle(d,d0)|≤10 (PCA参考)
  输出: 双答案 + 一致性 + 求值次数
"""
import math
import time

import numpy as np
import scipy.optimize as opt

import data_driven_axis as dd
import n2sjy2 as n2


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
    ad = math.degrees(math.acos(np.clip(np.dot(d, v), -1, 1)))
    ad1 = math.degrees(math.acos(np.clip(np.dot(d, d0), -1, 1)))
    return ad, ad1


def stage2_nsjy4(pts, r30, r45, a, v, d0, x0, max_evals=1000):
    evals = [0]

    def cost(x):
        evals[0] += 1
        ad, ad1 = angles_at(x, pts, r30, r45, a, v, d0)
        return abs(ad - ad1)

    def cb(xk):
        return cost(xk) <= 10.0

    bounds = [(-0.5, 0.5), (0.5, 3.0), (-0.5, 0.5), (0.5, 3.0)]
    res = opt.minimize(cost, np.array(x0, float), method="Nelder-Mead",
                       bounds=bounds, callback=cb,
                       options={"maxiter": max_evals, "maxfev": max_evals * 2})
    ad, ad1 = angles_at(res.x, pts, r30, r45, a, v, d0)
    return res.x, evals[0], abs(ad - ad1), ad, ad1


def angle(u, vv):
    return math.degrees(math.acos(np.clip(np.dot(u, vv), -1, 1)))


def main():
    datasets = {"cube0": dd.make_cube(10), "cube1": dd.make_cube(11),
                "ball0": dd.make_ball(0), "ball1": dd.make_ball(1),
                "ellip0": dd.make_ellip(0), "ellip1": dd.make_ellip(1)}
    print("=" * 118)
    print("融合算法: 阶段1 n2sjy2独立漂移(无PCA) → 阶段2 nsjy4热启动(PCA参考) → 双答案")
    print("=" * 118)
    print(f"{'name':8s} {'∠(d0,v)':>7s} | {'∠(d_n2,v)':>8s} {'∠(d_fast,v)':>10s} "
          f"{'∠(d_n2,dfast)':>12s} | {'阶段2evals':>10s} {'|Δ|fast':>7s} {'达标':>4s}")
    for name, pts in datasets.items():
        r30, r45, a, v, d0 = setup(pts)
        # 阶段1: n2sjy2 独立漂移 (无PCA)
        t0 = time.time()
        d_n2, t1n, t2n, _, iters = dd.data_driven_axis(pts, d0)
        # 阶段2: nsjy4 热启动
        x_f, ev, delf, ad, ad1 = stage2_nsjy4(pts, r30, r45, a, v, d0,
                                              [t1n.real, t1n.imag, t2n.real, t2n.imag])
        t1f = complex(x_f[0], x_f[1])
        t2f = complex(x_f[2], x_f[3])
        # 阶段2 的方向
        pt, p1, p2, sig = dd.fast_probs(r30, r45, t1f, t2f, a)
        F1f, F2f = n2.extract_foci(pts, p1, sig[1], p2, sig[2])
        d_fast = np.array(F1f, float) - np.array(F2f, float)
        d_fast = d_fast / np.linalg.norm(d_fast)
        ok = "✓" if delf <= 10 else "✗"
        print(f"{name:8s} {angle(d0, v):7.1f} | {angle(d_n2, v):8.1f} {angle(d_fast, v):10.1f} "
              f"{angle(d_n2, d_fast):12.1f} | {ev:10d} {delf:7.2f} {ok:>4s} "
              f"[阶段1={iters}外层, {time.time()-t0:.0f}s]")
    print("DONE")


if __name__ == "__main__":
    main()
