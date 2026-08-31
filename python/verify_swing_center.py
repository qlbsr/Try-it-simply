# -*- coding: utf-8 -*-
"""
纯数据摆动中心: 无任何参考, 取 n2sjy2 方向轨迹的球形均值 (摆动中心)
  问题: 方向轨迹是否真的摆动? 摆动中心是否 ≈ PCA/真实轴?
"""
import math
import time

import numpy as np

import data_driven_axis as dd
import n2sjy2 as n2


def angle(u, v):
    return math.degrees(math.acos(np.clip(np.dot(u, v), -1, 1)))


def ellip_axis(seed):
    rng = np.random.default_rng(300 + seed)
    u = rng.standard_normal(3)
    return u / np.linalg.norm(u)


def drift_directions(pts, rounds=40, n_inner=15, omega=0.9):
    rp = n2.compute_rp(pts)
    d2 = math.asin(math.cos(math.radians(30)) / math.pi)
    a = rp * (1 + math.sin(d2))
    _, r45, r30, _ = n2.yzqx(pts, rp)
    t1t, t2t = n2.compute_taus()
    pt0, p10, p20, sig0 = dd.fast_probs(r30, r45, t1t, t2t, a)
    F10, F20 = n2.extract_foci(pts, p10, sig0[1], p20, sig0[2])
    d0 = np.array(F10, float) - np.array(F20, float)
    d0 = d0 / np.linalg.norm(d0)
    t1 = t2 = complex(0, 1)
    d = d0.copy()
    dirs = [d.copy()]
    for k in range(rounds):
        t1b, t2b, F1, F2 = dd.inner_refine(t1, t2, pts, r30, r45, a, d, n_inner)
        pt, p1, p2, sig = dd.fast_probs(r30, r45, t1b, t2b, a)
        F1f, F2f = n2.extract_foci(pts, p1, sig[1], p2, sig[2])
        d_new = np.array(F1f, float) - np.array(F2f, float)
        d_new = d_new / np.linalg.norm(d_new)
        dirs.append(d_new.copy())
        d = d + omega * (d_new - d)
        d = d / np.linalg.norm(d)
        t1, t2 = t1b, t2b
    D = np.array(dirs)
    # 球形均值 (归一化向量和)
    mean_d = D.mean(axis=0)
    mean_d = mean_d / np.linalg.norm(mean_d)
    # 摆动范围: 轨迹内最大两两夹角
    max_swing = max(angle(D[i], D[j]) for i in range(len(D)) for j in range(i + 1, len(D)))
    # 后期稳定性: 后10点与均值的最大夹角
    tail = max(angle(mean_d, D[i]) for i in range(len(D) - 10, len(D)))
    return mean_d, max_swing, tail, D


def main():
    datasets = {"cube0": dd.make_cube(10), "ball0": dd.make_ball(0),
                "ellip0": dd.make_ellip(0), "ellip1": dd.make_ellip(1)}
    rng = np.random.default_rng(1000 + 5)
    datasets["gauss5"] = [rng.standard_normal(3) * (0.5 + 2 * rng.random()) for _ in range(200)]

    print("=" * 100)
    print("纯数据摆动中心: 方向轨迹球形均值 (无任何参考)")
    print("=" * 100)
    print(f"{'name':8s} {'摆动范围':>8s} {'∠(均值,v)':>8s} {'∠(均值,u)':>8s} {'后期稳定':>8s}")
    for name, pts in datasets.items():
        t0 = time.time()
        mean_d, swing, tail, D = drift_directions(pts, rounds=40)
        v = n2.pca_axis(pts)
        au = float("nan")
        if name.startswith("ellip"):
            au = angle(mean_d, ellip_axis(int(name[-1])))
        print(f"{name:8s} {swing:8.1f} {angle(mean_d, v):8.1f} {au:8.1f} {tail:8.2f}  "
              f"[{time.time()-t0:.0f}s]", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
