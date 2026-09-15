# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
确认: 固定数据轴 bis = normalize(d0 + d_i) (d_i=方形格参考) 全数据集收敛性
  无PCA: 内层LM(轴=bis) → 自由重拟合 → |Δ'|(固定参考) ≤10?
"""
import math
import time

import numpy as np

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
    pti, p1i, p2i, sigi = dd.fast_probs(r30, r45, complex(0, 1), complex(0, 1), a)
    F1i, F2i = n2.extract_foci(pts, p1i, sigi[1], p2i, sigi[2])
    d_i = np.array(F1i, float) - np.array(F2i, float)
    d_i = d_i / np.linalg.norm(d_i)
    return r30, r45, a, v, d0, d_i


def fixed_data_bisector(pts, rounds=30, n_inner=15):
    r30, r45, a, v, d0, d_i = setup(pts)
    bis = d0 + d_i
    bis = bis / np.linalg.norm(bis)
    t1 = t2 = complex(0, 1)
    deltas = []
    dirs = []
    for k in range(rounds):
        t1b, t2b, F1, F2 = dd.inner_refine(t1, t2, pts, r30, r45, a, bis, n_inner)
        pt, p1, p2, sig = dd.fast_probs(r30, r45, t1b, t2b, a)
        F1f, F2f = n2.extract_foci(pts, p1, sig[1], p2, sig[2])
        d_new = np.array(F1f, float) - np.array(F2f, float)
        d_new = d_new / np.linalg.norm(d_new)
        deltas.append(abs(angle(d_new, d0) - angle(d_new, d_i)))
        dirs.append(d_new.copy())
        t1, t2 = t1b, t2b
    return np.array(deltas), dirs[-1], bis


def main():
    datasets = {"cube0": dd.make_cube(10), "cube1": dd.make_cube(11),
                "ball0": dd.make_ball(0), "ball1": dd.make_ball(1),
                "ellip0": dd.make_ellip(0), "ellip1": dd.make_ellip(1)}
    rng = np.random.default_rng(1000 + 5)
    datasets["gauss5"] = [rng.standard_normal(3) * (0.5 + 2 * rng.random()) for _ in range(200)]

    print("=" * 96)
    print("无PCA 固定数据轴 bis=normalize(d0+d_i): 30轮 |Δ'|(固定参考) 是否≤10")
    print("=" * 96)
    print(f"{'name':8s} {'min|Δ\'|':>8s} {'max|Δ\'|':>8s} {'≤10':>4s} {'∠(终d,v)':>8s}")
    for name, pts in datasets.items():
        t0 = time.time()
        deltas, dfinal, bis = fixed_data_bisector(pts)
        r30, r45, a, v, d0, d_i = setup(pts)
        ok = "✓" if deltas.max() <= 10 else "✗"
        print(f"{name:8s} {deltas.min():8.2f} {deltas.max():8.2f} {ok:>4s} "
              f"{angle(dfinal, v):8.1f}  [{time.time()-t0:.0f}s]", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
