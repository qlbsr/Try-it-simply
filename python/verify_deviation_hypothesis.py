# -*- coding: utf-8 -*-
"""
验证用户猜想:
  "角度偏差与 (t1,t2) 偏离 (0,1) 的距离 / 对称性成正比" → 可用固定比例修正项迭代, 无需 pcav3.

E1. 固定参考点 (0,1)=(i,i) 与理论初值: 若"靠近(0,1)⇒角度小"成立, 各集 θ(i,i) 应都小
E2. 每个数据集内随机 τ 扫描: θ 与 D1=|τ1-i|+|τ2-i|、D2=|τ1-τ2| 的 Spearman 相关
E3. 沿射线 τ(s)=((0,1+s),(0,1+s)): θ(s) 是否单调
E4. 跨数据集: 同一 τ 的 θ 是否一致 (若 θ 仅由 τ 偏差决定, 应一致)
"""
import math
import time

import numpy as np

import nsjy_algorithms as m


def ang_deg(u, v):
    d = abs(float(np.dot(u, v)))
    return float(np.degrees(np.arccos(np.clip(d, 0.0, 1.0))))


def spearman(a, b):
    ra = np.argsort(np.argsort(np.asarray(a))).astype(float)
    rb = np.argsort(np.argsort(np.asarray(b))).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def prepare(pts):
    rp = m.compute_rp(pts)
    cone = math.asin(math.cos(math.radians(30.0)) / math.pi)
    a = rp * (1.0 + math.sin(cone))
    r30 = [m.inverse_th4(p, 30, rp) for p in pts]
    r45 = [m.inverse_th4(p, 45, rp) for p in pts]
    pc1, v3, ex = m.pca(pts)
    axis = pc1 / np.linalg.norm(pc1)
    return r30, r45, a, axis


def direction(pts, r30, r45, a, t1, t2):
    pt, p1, p2, sigm = m.compute_probabilities_from_taus(r30, r45, t1, t2, a)
    F1, F2 = m.extract_foci(pts, p1, sigm[1], p2, sigm[2])
    d = F1 - F2
    nrm = np.linalg.norm(d)
    return (F1, F2, d / nrm) if nrm > 1e-9 else (F1, F2, None)


def make_ellipsoid_points(u, n, seed, a0=1.5, b0=0.8, noise=0.02):
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


def build_sets():
    sets = {}
    for s in range(5):
        sets[f"ball{s}"] = m.random_points(200, seed=s)
    for s in range(5):
        rng = np.random.default_rng(100 + s)
        sets[f"gauss{s}"] = [rng.standard_normal(3) * (0.5 + 2.0 * rng.random()) for _ in range(200)]
    for s in range(5):
        rng = np.random.default_rng(200 + s)
        u = rng.standard_normal(3)
        u = u / np.linalg.norm(u)
        sets[f"ellip{s}"] = make_ellipsoid_points(u, 200, s, noise=0.02)
    return sets


def main():
    sets = build_sets()
    prep = {k: prepare(v) for k, v in sets.items()}
    t10, t20 = m.compute_taus()
    names = list(sets.keys())

    print("=" * 100)
    print("E1. 固定参考点: θ(i,i) 与 θ(theory)。若'靠近(0,1)⇒角度小', θ(i,i) 应全小")
    print("=" * 100)
    for name in names:
        r30, r45, a, axis = prep[name]
        _, _, dn = direction(sets[name], r30, r45, a, 1j, 1j)
        _, _, dn2 = direction(sets[name], r30, r45, a, t10, t20)
        th_ii = ang_deg(dn, axis) if dn is not None else float("nan")
        th_th = ang_deg(dn2, axis) if dn2 is not None else float("nan")
        print(f"{name:8s} θ(i,i)={th_ii:6.1f}°   θ(theory)={th_th:6.1f}°")

    print()
    print("=" * 100)
    print("E2. 数据集内随机 τ 扫描 (30 个): Spearman 相关")
    print("     D1 = |τ1-i|+|τ2-i| (偏离(0,1)),  D2 = |τ1-τ2| (不对称度)")
    print("=" * 100)
    sel = ["ball0", "ball2", "ball4", "gauss0", "gauss2", "ellip0"]
    rng = np.random.default_rng(42)
    pool_th, pool_d1 = [], []
    for name in sel:
        r30, r45, a, axis = prep[name]
        ths, d1s, d2s = [], [], []
        for _ in range(30):
            t1 = complex(rng.uniform(-0.5, 0.5), rng.uniform(0.8, 2.5))
            t2 = complex(rng.uniform(-0.5, 0.5), rng.uniform(0.8, 2.5))
            _, _, dn = direction(sets[name], r30, r45, a, t1, t2)
            if dn is None:
                continue
            th = ang_deg(dn, axis)
            ths.append(th)
            d1s.append(abs(t1 - 1j) + abs(t2 - 1j))
            d2s.append(abs(t1 - t2))
        rho1 = spearman(ths, d1s)
        rho2 = spearman(ths, d2s)
        pool_th += ths
        pool_d1 += d1s
        print(f"{name:8s} n={len(ths):2d}  ρ(θ,D1)={rho1:+.2f}   ρ(θ,D2)={rho2:+.2f}")
    print(f"合并(所有集): ρ(θ,D1)={spearman(pool_th, pool_d1):+.2f}  "
          f"(若普适正比, 应接近 +1 且各集同号)")

    print()
    print("=" * 100)
    print("E3. 沿射线 τ(s)=((0,1+s),(0,1+s)), s=0..1.5: θ(s) 应单调递减/递增?")
    print("=" * 100)
    for name in ("ball0", "gauss0", "ellip0"):
        r30, r45, a, axis = prep[name]
        vals = []
        for s in np.linspace(0.0, 1.5, 16):
            t = complex(0.0, 1.0 + s)
            _, _, dn = direction(sets[name], r30, r45, a, t, t)
            vals.append(ang_deg(dn, axis) if dn is not None else float("nan"))
        print(f"{name:8s} θ(s)=" + " ".join(f"{v:3.0f}" for v in vals))

    print()
    print("DONE")


if __name__ == "__main__":
    main()
