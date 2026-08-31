# -*- coding: utf-8 -*-
"""
完全脱离 PCA 验证:
  参考对 (d0, d_i): d0=理论τ焦点方向, d_i=方形格(0,1),(0,1)焦点方向   ← 纯数据, 无PCA
  1. n2sjy2 效果: 无PCA漂移的方向轨迹 + |Δ'|=|angle(d,d0)-angle(d,d_i)| 摆动
  2. 等分线 d* = normalize(d0+d_i)   ← 摆动回归的目标 (纯数据)
  3. NM 无PCA优化 |Δ'|≤10 → 最终方向
  诊断: ∠(最终d, v) 仅事后看相关性 (v 不进任何计算)
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
    v = n2.pca_axis(pts)                       # 仅事后诊断
    t1t, t2t = n2.compute_taus()
    pt0, p10, p20, sig0 = dd.fast_probs(r30, r45, t1t, t2t, a)
    F10, F20 = n2.extract_foci(pts, p10, sig0[1], p20, sig0[2])
    d0 = np.array(F10, float) - np.array(F20, float)
    d0 = d0 / np.linalg.norm(d0)
    # d_i: 方形格 (0,1),(0,1) 处的焦点方向 (无PCA参考)
    pti, p1i, p2i, sigi = dd.fast_probs(r30, r45, complex(0, 1), complex(0, 1), a)
    F1i, F2i = n2.extract_foci(pts, p1i, sigi[1], p2i, sigi[2])
    d_i = np.array(F1i, float) - np.array(F2i, float)
    d_i = d_i / np.linalg.norm(d_i)
    return r30, r45, a, v, d0, d_i


def direction_at(x, pts, r30, r45, a):
    t1 = complex(x[0], x[1])
    t2 = complex(x[2], x[3])
    pt, p1, p2, sig = dd.fast_probs(r30, r45, t1, t2, a)
    F1, F2 = n2.extract_foci(pts, p1, sig[1], p2, sig[2])
    d = np.array(F1, float) - np.array(F2, float)
    return d / np.linalg.norm(d)


def delta_prime(d, d0, d_i):
    return abs(angle(d, d0) - angle(d, d_i))


def drift_trajectory(pts, rounds=30, n_inner=15, omega=0.9):
    r30, r45, a, v, d0, d_i = setup(pts)
    t1 = t2 = complex(0, 1)
    d = d0.copy()
    traj = []
    for k in range(rounds):
        t1b, t2b, F1, F2 = dd.inner_refine(t1, t2, pts, r30, r45, a, d, n_inner)
        pt, p1, p2, sig = dd.fast_probs(r30, r45, t1b, t2b, a)
        F1f, F2f = n2.extract_foci(pts, p1, sig[1], p2, sig[2])
        d_new = np.array(F1f, float) - np.array(F2f, float)
        d_new = d_new / np.linalg.norm(d_new)
        traj.append((d_new.copy(), delta_prime(d_new, d0, d_i),
                     angle(d_new, d0), angle(d_new, d_i)))
        d = d + omega * (d_new - d)
        d = d / np.linalg.norm(d)
        t1, t2 = t1b, t2b
    return traj, d0, d_i, v


def nm_pca_free(pts, d0, d_i, x0, max_evals=400):
    r30, r45, a, v, _, _ = setup(pts)
    evals = [0]

    def cost(x):
        evals[0] += 1
        d = direction_at(x, pts, r30, r45, a)
        return delta_prime(d, d0, d_i)

    def cb(xk):
        return cost(xk) <= 10.0

    x0c = np.array([max(-0.5, min(0.5, x0[0])), max(0.5, min(3.0, x0[1])),
                    max(-0.5, min(0.5, x0[2])), max(0.5, min(3.0, x0[3]))])
    bounds = [(-0.5, 0.5), (0.5, 3.0), (-0.5, 0.5), (0.5, 3.0)]
    res = opt.minimize(cost, x0c, method="Nelder-Mead", bounds=bounds, callback=cb,
                       options={"maxiter": max_evals, "maxfev": max_evals * 2})
    return res.x, evals[0]


def main():
    datasets = {"cube0": dd.make_cube(10), "ball0": dd.make_ball(0),
                "ellip0": dd.make_ellip(0), "ellip1": dd.make_ellip(1)}
    rng = np.random.default_rng(1000 + 5)
    datasets["gauss5"] = [rng.standard_normal(3) * (0.5 + 2 * rng.random()) for _ in range(200)]

    print("=" * 118)
    print("完全脱离PCA: 参考(d0, d_i) 均数据驱动 | n2sjy2漂移摆动 | 等分线 | NM无PCA")
    print("=" * 118)
    print(f"{'name':8s} {'扇角∠(d0,di)':>11s} {'漂移|Δ\'|min':>11s} {'max':>6s} "
          f"{'等分线∠(d*,di)':>12s} {'NM后|Δ\'|':>8s} {'∠(终d,v)':>8s}")
    for name, pts in datasets.items():
        t0 = time.time()
        traj, d0, d_i, v = drift_trajectory(pts, rounds=30)
        deltas = np.array([t[1] for t in traj])
        dstar = d0 + d_i
        dstar = dstar / np.linalg.norm(dstar)
        # 最后 τ (从轨迹最后方向反推 — 用最后方向直接作为起点做NM)
        last_d = traj[-1][0]
        # 用最后方向无法直接转τ, 用轨迹最后的 (t1,t2) — 重新跑拿τ
        r30, r45, a, v2, d0b, d_ib = setup(pts)
        t1 = t2 = complex(0, 1)
        d = d0.copy()
        for k in range(30):
            t1b, t2b, F1, F2 = dd.inner_refine(t1, t2, pts, r30, r45, a, d, 15)
            pt, p1, p2, sig = dd.fast_probs(r30, r45, t1b, t2b, a)
            F1f, F2f = n2.extract_foci(pts, p1, sig[1], p2, sig[2])
            d_new = np.array(F1f, float) - np.array(F2f, float)
            d_new = d_new / np.linalg.norm(d_new)
            d = d + 0.9 * (d_new - d)
            d = d / np.linalg.norm(d)
            t1, t2 = t1b, t2b
        x0 = [t1.real, t1.imag, t2.real, t2.imag]
        xf, ev = nm_pca_free(pts, d0, d_i, x0)
        df = direction_at(xf, pts, r30, r45, a)
        delta_f = delta_prime(df, d0, d_i)
        print(f"{name:8s} {angle(d0, d_i):11.1f} {deltas.min():11.2f} {deltas.max():6.2f} "
              f"{angle(dstar, d_i):12.1f} {delta_f:8.2f} {angle(df, v):8.1f}  "
              f"[{time.time()-t0:.0f}s]", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
