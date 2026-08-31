# -*- coding: utf-8 -*-
"""
验证"概率作为修正项"的精确流程 (无 v3 参与更新, 不设 0.5° 死判据):
  初始概率集 → 拟合 F1,F2 → 用总概率加权修正 t1,t2 → 重新拟合 F1,F2 → 迭代
并检查:
  E0. 已跑批次的 θ_final 与 |t1-t2|(对称性)、D1(偏离(0,1)) 的关系 (含初始角)
  E1. 近邻点(如 gauss4 的 2.3°)是否处于跳变边界 (扰动灵敏性)
  E2. 概率修正环长跑: θ_line(0-90,取直线)、θ_or(0-180,保留方向) 是否收敛到固定角?
      固定角是否跨案例一致(容差下)? |t1-t2|、D1 轨迹如何?
"""
import math

import numpy as np

import nsjy_algorithms as m

# 之前批量测试的最终 (t1f,t2f, θ_final°) —— n=200
BATCH = {
    "ball0":   ((0.0000, 1.2793), (0.0000, 1.0000), 52.568),
    "ball1":   ((-0.0843, 1.6345), (-0.2413, 2.5950), 29.309),
    "ball2":   ((0.0813, 1.1921), (-0.2402, 1.0161), 0.471),
    "ball3":   ((-0.2696, 1.5967), (0.0241, 1.8914), 0.302),
    "ball4":   ((0.3792, 2.9749), (0.0760, 1.0551), 33.340),
    "gauss0":  ((0.0377, 1.1118), (-0.0127, 1.4561), 0.091),
    "gauss1":  ((0.0133, 1.2313), (0.0330, 1.1049), 0.119),
    "gauss2":  ((0.4576, 1.5203), (0.1973, 0.9804), 50.794),
    "gauss3":  ((0.2424, 1.8293), (0.1556, 1.0292), 0.498),
    "gauss4":  ((-0.4929, 3.4076), (0.3959, 4.3928), 2.305),
    "ellip0":  ((0.2307, 2.4625), (-0.2793, 1.9769), 0.045),
    "ellip1":  ((0.2483, 2.7106), (-0.0420, 1.7530), 21.958),
    "ellip2":  ((0.3035, 1.7386), (0.0509, 1.2829), 32.955),
    "ellip3":  ((0.1235, 1.6222), (0.0974, 0.9952), 37.342),
    "ellip4":  ((-0.1726, 2.9466), (-0.0479, 3.5594), 19.908),
    "ellipN0": ((0.3374, 1.4085), (0.1409, 3.0429), 17.232),
    "ellipN1": ((0.0000, 1.2793), (0.0000, 1.0000), 28.485),
    "ellipN2": ((-0.3984, 4.0544), (0.4241, 0.9056), 39.683),
    "ellipN3": ((0.2402, 1.3853), (-0.3740, 1.0859), 0.214),
    "ellipN4": ((-0.1279, 3.7821), (0.3630, 1.6878), 1.714),
}

# E1 实测的理论初值角 (15 个 ball/gauss/ellip)
TH_THEORY = {
    "ball0": 30.6, "ball1": 89.8, "ball2": 79.7, "ball3": 51.1, "ball4": 70.2,
    "gauss0": 51.7, "gauss1": 10.1, "gauss2": 13.5, "gauss3": 85.8, "gauss4": 37.3,
    "ellip0": 80.9, "ellip1": 59.1, "ellip2": 88.4, "ellip3": 78.5, "ellip4": 81.6,
}


def ang_line(d, axis):
    """0-90°, 直线方向(忽略±)"""
    dd = np.array(d, float)
    if np.dot(dd, axis) < 0:
        dd = -dd
    c = float(np.dot(dd, axis))
    return float(np.degrees(np.arccos(np.clip(c, 0.0, 1.0))))


def ang_or(d, axis):
    """0-180°, 保留方向"""
    c = float(np.dot(np.array(d, float), axis))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


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


def refit(pts, r30, r45, a, t1, t2):
    """初始概率集 → 拟合 F1,F2 (诊断用)"""
    pt, p1, p2, sigm = m.compute_probabilities_from_taus(r30, r45, t1, t2, a)
    F1, F2 = m.extract_foci(pts, p1, sigm[1], p2, sigm[2])
    d = F1 - F2
    nrm = np.linalg.norm(d)
    return pt, F1, F2, (d / nrm if nrm > 1e-9 else None)


def prob_weighted_update(pts, r30, r45, a, t1, t2, eta=0.3, rng=20):
    """根据总概率修正 t1,t2:
    τ ← τ + η·Σ_i w_i·n_i·(z_i − λ_i) / Σw ,  w_i = probTotal[i]
    (即 总概率加权的格拟合梯度: 把最近格点拉向数据点)"""
    pt, _, _, _ = refit(pts, r30, r45, a, t1, t2)
    lat1 = m.generate_lattice_1d(t1, rng)
    lat2 = m.generate_lattice_1d(t2, rng)
    mg = np.arange(-rng, rng + 1)
    MM, NN = np.meshgrid(mg, mg)
    Nflat = NN.ravel()
    g1 = 0j
    g2 = 0j
    W = 0.0
    for i, (z1, z2) in enumerate(zip(r30, r45)):
        w = float(pt[i])
        j1 = int(np.argmin(np.abs(z1 - lat1)))
        j2 = int(np.argmin(np.abs(z2 - lat2)))
        g1 += w * Nflat[j1] * (z1 - lat1[j1])
        g2 += w * Nflat[j2] * (z2 - lat2[j2])
        W += w
    g1 = g1 / max(W, 1e-9)
    g2 = g2 / max(W, 1e-9)
    step = abs(g1) + abs(g2)
    cap = 0.05
    if step > cap:
        g1 *= cap / step
        g2 *= cap / step
    return t1 + eta * g1, t2 + eta * g2


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


def build(name):
    if name.startswith("ball"):
        return m.random_points(200, seed=int(name[4:]))
    if name.startswith("gauss"):
        s = int(name[5:])
        rng = np.random.default_rng(100 + s)
        return [rng.standard_normal(3) * (0.5 + 2.0 * rng.random()) for _ in range(200)]
    if name.startswith("ellipN"):
        s = int(name[6:])
        rng = np.random.default_rng(300 + s)
        u = rng.standard_normal(3)
        u = u / np.linalg.norm(u)
        return make_ellipsoid_points(u, 200, s, noise=0.1)
    s = int(name[5:])
    rng = np.random.default_rng(200 + s)
    u = rng.standard_normal(3)
    u = u / np.linalg.norm(u)
    return make_ellipsoid_points(u, 200, s, noise=0.02)


def main():
    # ================= E0: 对称性关系表 =================
    print("=" * 100)
    print("E0. 已跑批次: θ_final 与 |t1-t2|(对称性), D1=|τ1-i|+|τ2-i|, 初始角 θ_theory")
    print("=" * 100)
    rows = []
    for name, ((r1, i1), (r2, i2), thf) in BATCH.items():
        t1 = complex(r1, i1)
        t2 = complex(r2, i2)
        d12 = abs(t1 - t2)
        D1 = abs(t1 - 1j) + abs(t2 - 1j)
        th0 = TH_THEORY.get(name, float("nan"))
        rows.append((name, th0, thf, d12, D1))
    rows.sort(key=lambda r: r[2])
    print(f"{'name':8s} {'θ理论初':>7s} {'θ_final':>8s} {'|t1-t2|':>8s} {'D1':>7s}  说明")
    for name, th0, thf, d12, D1 in rows:
        tag = "收敛" if thf < 0.5 else ("近邻" if thf < 5 else "未收敛")
        print(f"{name:8s} {th0:7.1f} {thf:8.2f} {d12:8.3f} {D1:7.3f}  {tag}")
    ths = [r[2] for r in rows]
    d12s = [r[3] for r in rows]
    D1s = [r[4] for r in rows]
    print(f"ρ(θ_final, |t1-t2|) = {spearman(ths, d12s):+.2f}    "
          f"ρ(θ_final, D1) = {spearman(ths, D1s):+.2f}")
    print("反例: ball0 偏差最小(0.279)却 52.6°; gauss4 偏差最大(1.33)却 2.3° → 无单调关系")

    # ================= E1: 近邻点扰动灵敏性 =================
    print()
    print("=" * 100)
    print("E1. 扰动灵敏性 (判断 2.3° 是否处于折痕跳变边界)")
    print("=" * 100)
    for name in ("gauss4", "ball0", "ellipN4"):
        pts = build(name)
        r30, r45, a, axis = prepare(pts)
        (r1, i1), (r2, i2), _ = BATCH[name]
        rng = np.random.default_rng(3)
        angs = []
        for _ in range(20):
            amp = rng.uniform(0.02, 0.12)
            ph = rng.uniform(0.0, 2.0 * np.pi)
            t1p = complex(r1, i1) + amp * complex(math.cos(ph), math.sin(ph))
            ph = rng.uniform(0.0, 2.0 * np.pi)
            t2p = complex(r2, i2) + amp * complex(math.cos(ph), math.sin(ph))
            _, _, _, dn = refit(pts, r30, r45, a, t1p, t2p)
            if dn is not None:
                angs.append(ang_line(dn, axis))
        print(f"{name:8s} 20次扰动: θ min={min(angs):5.1f}°  max={max(angs):5.1f}°  "
              f"spread={max(angs)-min(angs):5.1f}°  (原值 {BATCH[name][2]:.1f}°)")

    # ================= E2: 概率修正环 (无 v3) =================
    print()
    print("=" * 100)
    print("E2. 总概率加权修正环 150 次 (无 v3, 每轮重拟合 F1,F2)")
    print("    θ_line(0-90,取直线)  θ_or(0-180,保留方向)  |t1-t2|  D1")
    print("=" * 100)
    t10, t20 = m.compute_taus()
    for name in ("ball0", "gauss2", "gauss4", "ellip0"):
        pts = build(name)
        r30, r45, a, axis = prepare(pts)
        t1, t2 = t10, t20
        for it in range(151):
            pt, F1, F2, dn = refit(pts, r30, r45, a, t1, t2)
            if dn is not None and it % 25 == 0:
                print(f"{name:8s} it={it:3d}  θ_line={ang_line(dn, axis):6.1f}°  "
                      f"θ_or={ang_or(dn, axis):6.1f}°  |t1-t2|={abs(t1-t2):6.3f}  "
                      f"D1={abs(t1-1j)+abs(t2-1j):6.3f}  t1=({t1.real:.3f},{t1.imag:.3f}) "
                      f"t2=({t2.real:.3f},{t2.imag:.3f})")
            t1, t2 = prob_weighted_update(pts, r30, r45, a, t1, t2)
        print("-" * 100)

    print("DONE")


if __name__ == "__main__":
    main()
