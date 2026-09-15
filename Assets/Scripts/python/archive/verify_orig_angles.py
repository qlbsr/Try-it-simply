# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""列出全部数据集的 原始角(理论初值处θ) 与 PCA角(环终点θ) 对比"""
import numpy as np

import nsjy_algorithms as m
import verify_probability_correction as vp

t10, t20 = m.compute_taus()

THETA_PCA = {
    "ball0": 52.568, "ball1": 29.309, "ball2": 0.471, "ball3": 0.302, "ball4": 33.340,
    "gauss0": 0.091, "gauss1": 0.119, "gauss2": 50.794, "gauss3": 0.498, "gauss4": 2.305,
    "ellip0": 0.045, "ellip1": 21.958, "ellip2": 32.955, "ellip3": 37.342, "ellip4": 19.908,
    "ellipN0": 17.232, "ellipN1": 28.485, "ellipN2": 39.683, "ellipN3": 0.214, "ellipN4": 1.714,
    "ball5": 29.9, "ball6": 42.0, "ball7": 42.8, "ball8": 0.1,
    "gauss5": 0.1, "gauss6": 15.3, "gauss7": 0.1, "gauss8": 23.6,
    "ellip5": 0.1, "ellip6": 0.1, "ellip7": 17.4,
    "ellipN5": 0.0, "ellipN6": 0.2, "ellipN7": 1.9,
}


def make_set(tag, seed, n=200):
    if tag == "ball":
        return m.random_points(n, seed=seed)
    if tag == "gauss":
        rng = np.random.default_rng(1000 + seed)
        return [rng.standard_normal(3) * (0.5 + 2.0 * rng.random()) for _ in range(n)]
    if tag == "ellip":
        rng = np.random.default_rng(2000 + seed)
        u = rng.standard_normal(3)
        u = u / np.linalg.norm(u)
        return vp.make_ellipsoid_points(u, n, 1000 + seed, noise=0.02)
    if tag == "ellipN":
        rng = np.random.default_rng(3000 + seed)
        u = rng.standard_normal(3)
        u = u / np.linalg.norm(u)
        return vp.make_ellipsoid_points(u, n, 1000 + seed, noise=0.1)
    raise ValueError(tag)


def get_pts(name):
    if name in vp.BATCH:
        return vp.build(name)
    tag = name.rstrip("0123456789")
    seed = int(name[len(tag):])
    return make_set(tag, seed)


names = list(THETA_PCA.keys())
rows = []
for name in names:
    pts = get_pts(name)
    r30, r45, a, axis = vp.prepare(pts)
    _, _, _, dn = vp.refit(pts, r30, r45, a, t10, t20)
    th_orig = vp.ang_line(dn, axis) if dn is not None else float("nan")
    rows.append((name, th_orig, THETA_PCA[name]))

rows.sort(key=lambda r: r[1])
print(f"{'name':8s} {'原始角θ0':>8s} {'PCA角θpca':>9s} {'θpca-θ0':>8s}  档位")
for name, th0, thp in rows:
    diff = thp - th0
    tier = "档1成立" if thp < 0.5 else ("容错带" if thp <= 30 else "档2准入")
    print(f"{name:8s} {th0:8.1f} {thp:9.1f} {diff:+8.1f}  {tier}")
print()
inc = [(x[0], x[1], x[2]) for x in rows if x[2] - x[1] > 1.0]
print(f"PCA角比原始角增大>1°的案例: {len(inc)} 个 -> "
      + ", ".join(f"{n}({a:.0f}→{b:.0f})" for n, a, b in inc))
print("DONE")
