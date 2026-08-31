# -*- coding: utf-8 -*-
"""
设计B: 内层轴固定为等分大圆方向 d* = normalize(v+d0) (而非漂移的d)
  每轮: 内层LM(轴=d*) → 自由重拟合 → |Δ|
  验证: 30 轮 7 数据集 |Δ| 是否全程≤10 (构造性自持, 不需要nsjy4点火)
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


def fixed_axis_run(pts, rounds=30, n_inner=15):
    r30, r45, a, v, d0 = setup(pts)
    # 固定轴 = 等分大圆方向 (|Δ|=0 的解析点)
    dstar = v + d0
    dstar = dstar / np.linalg.norm(dstar)
    t1 = t2 = complex(0, 1)
    deltas = []
    for k in range(rounds):
        t1b, t2b, F1, F2 = dd.inner_refine(t1, t2, pts, r30, r45, a, dstar, n_inner)
        xb = [t1b.real, t1b.imag, t2b.real, t2b.imag]
        ad, ad1 = angles_at(xb, pts, r30, r45, a, v, d0)
        deltas.append(abs(ad - ad1))
        t1, t2 = t1b, t2b
    return np.array(deltas), dstar


def main():
    datasets = {
        "cube0": dd.make_cube(10), "cube1": dd.make_cube(11),
        "ball0": dd.make_ball(0), "ball1": dd.make_ball(1),
        "ellip0": dd.make_ellip(0), "ellip1": dd.make_ellip(1),
    }
    rng = np.random.default_rng(1000 + 5)
    datasets["gauss5"] = [rng.standard_normal(3) * (0.5 + 2 * rng.random()) for _ in range(200)]

    print("=" * 96)
    print("设计B: 内层轴=固定大圆方向 d*=normalize(v+d0), 30轮, 无nsjy4")
    print("=" * 96)
    print(f"{'name':8s} {'全程min|Δ|':>9s} {'全程max|Δ|':>9s} {'全程≤10':>7s} "
          f"{'∠(d*,v)':>8s} {'∠(d0,v)':>7s}")
    for name, pts in datasets.items():
        t0 = time.time()
        deltas, dstar = fixed_axis_run(pts)
        r30, r45, a, v, d0 = setup(pts)
        ok = "✓" if deltas.max() <= 10 else "✗"
        print(f"{name:8s} {deltas.min():9.2f} {deltas.max():9.2f} {ok:>7s} "
              f"{angle(dstar, v):8.1f} {angle(d0, v):7.1f}  [{time.time()-t0:.0f}s]", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
