# -*- coding: utf-8 -*-
"""
闭环收敛批量测试: 多种随机点集 → 判断能否跑通 / 收敛正确。
用法: python test_convergence.py
"""
import sys
import time

import numpy as np

import nsjy_algorithms as m


def make_ellipsoid_points(u, n, seed, a0=1.5, b0=0.8, noise=0.02):
    """沿已知轴 u 的椭球面 + 小噪声, 用于验证闭环收敛到正确轴"""
    rng = np.random.default_rng(seed)
    u = np.asarray(u, float)
    u = u / np.linalg.norm(u)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    v = np.cross(u, tmp)
    v = v / np.linalg.norm(v)
    w = np.cross(u, v)
    th = rng.uniform(0.0, np.pi, n)
    ph = rng.uniform(0.0, 2.0 * np.pi, n)
    pts = []
    for i in range(n):
        p = (a0 * np.sin(th[i]) * np.cos(ph[i]) * u
             + b0 * np.sin(th[i]) * np.sin(ph[i]) * v
             + b0 * np.cos(th[i]) * w)
        pts.append(p + noise * rng.standard_normal(3))
    return pts


def run_case(name, points, max_iter=40, angle_tol=0.5):
    t0 = time.time()
    try:
        t10, t20 = m.compute_taus()
        pc1, v3, ex = m.pca(points)
        axis = pc1 / np.linalg.norm(pc1)
        t1f, t2f, F1f, F2f = m.refine_moduli_by_axis(
            points, t10, t20, axis,
            max_iter=max_iter, angle_tol_deg=angle_tol, verbose=False)
        d = F1f - F2f
        dn = d / np.linalg.norm(d)
        dotv = abs(float(np.dot(dn, axis)))
        ang = float(np.degrees(np.arccos(np.clip(dotv, -1.0, 1.0))))
        ok = ang < angle_tol
        dt = time.time() - t0
        print(f"{name:30s} {'CONVERGED' if ok else 'NOT-CONV':10s} "
              f"ang={ang:7.3f}° dot={dotv:.4f} "
              f"t1f=({t1f.real:.4f},{t1f.imag:.4f}) t2f=({t2f.real:.4f},{t2f.imag:.4f}) "
              f"{dt:5.1f}s", flush=True)
    except Exception as e:
        dt = time.time() - t0
        print(f"{name:30s} {'EXCEPTION':10s} {type(e).__name__}: {e} ({dt:.1f}s)", flush=True)


print("=== 1) 单位球均匀随机点 (不同 seed) ===")
for seed in range(5):
    run_case(f"ball seed={seed}", m.random_points(200, seed=seed))

print("=== 2) 高斯散布随机点 (不同 seed, 尺度随机) ===")
for seed in range(5):
    rng = np.random.default_rng(100 + seed)
    pts = [rng.standard_normal(3) * (0.5 + 2.0 * rng.random()) for _ in range(200)]
    run_case(f"gauss seed={seed}", pts)

print("=== 3) 椭球面点(已知轴) 噪声0.02 ===")
for seed in range(5):
    rng = np.random.default_rng(200 + seed)
    u = rng.standard_normal(3)
    u = u / np.linalg.norm(u)
    run_case(f"ellip seed={seed} u={u.round(2)}",
             make_ellipsoid_points(u, 200, seed, noise=0.02))

print("=== 4) 椭球面点 噪声0.1 (更强噪声) ===")
for seed in range(5):
    rng = np.random.default_rng(300 + seed)
    u = rng.standard_normal(3)
    u = u / np.linalg.norm(u)
    run_case(f"ellipN seed={seed} u={u.round(2)}",
             make_ellipsoid_points(u, 200, seed, noise=0.1))

print("DONE")
