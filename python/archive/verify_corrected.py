# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""修正循环后 (标准模(0,1),(0,1) 重试 + 重试门) 重跑 34 个数据集, 生成新档位表"""
import numpy as np

import nsjy_algorithms as m
import verify_probability_correction as vp

t10, t20 = m.compute_taus()

OLD_PCA = {
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


names = list(OLD_PCA.keys())
rows = []
for name in names:
    pts = get_pts(name)
    r30, r45, a, axis = vp.prepare(pts)
    _, _, _, dn0 = vp.refit(pts, r30, r45, a, t10, t20)
    th_orig = vp.ang_line(dn0, axis) if dn0 is not None else float("nan")
    t1f, t2f, F1f, F2f = m.refine_moduli_by_axis(pts, t10, t20, axis,
                                                 max_iter=40, verbose=False)
    d = F1f - F2f
    dn = d / np.linalg.norm(d)
    th_new = vp.ang_line(dn, axis)
    rows.append((name, th_orig, OLD_PCA[name], th_new, t1f, t2f))

rows.sort(key=lambda r: r[0])
print(f"{'name':8s} {'原始角':>7s} {'旧PCA角':>7s} {'新PCA角':>7s} {'变化':>6s}  档位(新)")
changed = 0
for name, th0, th_old, th_new, t1f, t2f in rows:
    diff = th_new - th_old
    tier = "档1成立" if th_new < 0.5 else ("容错带" if th_new <= 30 else "档2")
    flag = "  <== 变化" if abs(diff) > 0.5 else ""
    if abs(diff) > 0.5:
        changed += 1
    print(f"{name:8s} {th0:7.1f} {th_old:7.1f} {th_new:7.1f} {diff:+6.1f}  {tier}{flag}")
print()
print(f"θ_PCA 变化>0.5° 的案例: {changed} 个")
n1 = sum(1 for r in rows if r[3] < 0.5)
nm = sum(1 for r in rows if 0.5 <= r[3] <= 30)
n2 = sum(1 for r in rows if r[3] > 30)
print(f"新档位: 档1={n1}  容错带={nm}  档2={n2}")
print("DONE")
