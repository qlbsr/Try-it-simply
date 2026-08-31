# -*- coding: utf-8 -*-
"""全部 52 点集: 自迭代稳定后的 max|probTotal-probFoci| 与 RMS 分布"""
import numpy as np

import nsjy_algorithms as m
import verify_probability_correction as vp

t10, t20 = m.compute_taus()


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


def plateau_dev(pts, r30, r45, a, t1, t2, n_iter=120, every=40):
    Z1 = np.asarray(r30, complex)
    Z2 = np.asarray(r45, complex)
    P = np.asarray(pts, float)
    mg = np.arange(-20, 21)
    MM, NN = np.meshgrid(mg, mg)
    MC, NC = MM.ravel(), NN.ravel()
    md_last = rms_last = None
    for it in range(n_iter + 1):
        L1 = MC + NC * t1
        L2 = MC + NC * t2
        D1 = np.abs(Z1[:, None] - L1[None, :])
        j1 = np.argmin(D1, axis=1)
        d1 = D1[np.arange(len(Z1)), j1]
        D2 = np.abs(Z2[:, None] - L2[None, :])
        j2 = np.argmin(D2, axis=1)
        d2 = D2[np.arange(len(Z2)), j2]
        s1 = np.sqrt(np.mean(d1 * d1))
        s2 = np.sqrt(np.mean(d2 * d2))
        p1 = np.exp(-d1 * d1 / (2 * s1 * s1))
        p2 = np.exp(-d2 * d2 / (2 * s2 * s2))
        pt = np.exp(-np.abs(d1 + d2 - 2 * a) / (2 * a))
        p0 = P[0]
        A = 2.0 * (P[1:] - p0)
        b = (P[1:] ** 2).sum(1) - (p0 ** 2).sum() - (d1[1:] ** 2 - d1[0] ** 2)
        F1, *_ = np.linalg.lstsq(A, b, rcond=None)
        b2 = (P[1:] ** 2).sum(1) - (p0 ** 2).sum() - (d2[1:] ** 2 - d2[0] ** 2)
        F2, *_ = np.linalg.lstsq(A, b2, rcond=None)
        pf = np.array([np.exp(-abs(np.linalg.norm(p - F1) + np.linalg.norm(p - F2) - 2 * a) / (2 * a))
                       for p in P])
        maxdev = float(np.max(np.abs(pt - pf)))
        rms = float(np.sqrt(np.mean((pt - pf) ** 2)))
        if it % every == 0:
            md_last, rms_last = maxdev, rms
        w = pt
        W = w.sum() if w.sum() > 1e-9 else 1.0
        g1 = np.sum(w * NC[j1] * (Z1 - L1[j1])) / W
        g2 = np.sum(w * NC[j2] * (Z2 - L2[j2])) / W
        st = abs(g1) + abs(g2)
        cap = 0.05
        if st > cap:
            g1 *= cap / st
            g2 *= cap / st
        t1 = t1 + 0.3 * g1
        t2 = t2 + 0.3 * g2
    return md_last, rms_last


names = ([f"ball{i}" for i in range(14)] + [f"gauss{i}" for i in range(14)]
         + [f"ellip{i}" for i in range(12)] + [f"ellipN{i}" for i in range(12)])

rows = []
for name in names:
    pts = get_pts(name)
    r30, r45, a, axis = vp.prepare(pts)
    md, rms = plateau_dev(pts, r30, r45, a, t10, t20)
    rows.append((name, md, rms))

md = np.array([r[1] for r in rows])
rms = np.array([r[2] for r in rows])

print(f"{'name':8s} {'maxDev':>7s} {'RMS':>6s}")
for name, mdv, rmsv in sorted(rows, key=lambda r: -r[1]):
    print(f"{name:8s} {mdv:7.3f} {rmsv:6.3f}")

print()
print("=" * 60)
print("max|probTotal-probFoci| (地板) 分布")
print("=" * 60)
print(f"min={md.min():.3f}  mean={md.mean():.3f}  max={md.max():.3f}  std={md.std():.3f}")
print(f"RMS 偏差: min={rms.min():.3f}  mean={rms.mean():.3f}  max={rms.max():.3f}")
hist = {0.4: 0, 0.5: 0, 0.6: 0, 0.7: 0, 99: 0}
for v in md:
    for k in (0.4, 0.5, 0.6, 0.7, 99):
        if v < k:
            hist[k] += 1
            break
print("maxDev 桶分布 (每桶计数):")
labels = [(0.4, "<0.4"), (0.5, "0.4~0.5"), (0.6, "0.5~0.6"), (0.7, "0.6~0.7"), (99, "≥0.7")]
for k, lab in labels:
    print(f"  {lab:>8s}: {hist[k]:3d}")
print("DONE")
