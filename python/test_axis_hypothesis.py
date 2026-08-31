# -*- coding: utf-8 -*-
"""
假设验证: 是否存在对所有点集固定的 (t1,t2)?
          去掉 pcav3 标准轴后, 仅靠数据自洽能否自我收敛并靠近 v3?

实验:
  P1. 方向敏感性: 同一数据集, 不同 τ → d(τ)=(F1-F2)方向 是否变化? 与 v3 夹角?
  P2. 固定 τ 普适性: 用某集收敛出的 (t1f,t2f) 去算其他集的 d, 是否都贴合各自 v3?
  P3. 去掉 PCA 的自洽环: wDir=0 跑闭环, 看最终方向是否靠近 v3;
      并比较 自洽代价 在 理论τ / PCA收敛τ / 自洽环结果 三处的取值.
  P4. 椭圆残差准则: 若用 椭球残差 Σ(|p-F1|+|p-F2|-2a)² 选 τ, 最优 τ 的方向是否即 v3?
"""
import math
import time

import numpy as np

import nsjy_algorithms as m


def ang_deg(u, v):
    """两单位向量的夹角(取直线方向, 忽略 ±)"""
    d = abs(float(np.dot(u, v)))
    return float(np.degrees(np.arccos(np.clip(d, 0.0, 1.0))))


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
    """τ → (F1, F2, 单位方向)"""
    pt, p1, p2, sigm = m.compute_probabilities_from_taus(r30, r45, t1, t2, a)
    F1, F2 = m.extract_foci(pts, p1, sigm[1], p2, sigm[2])
    d = F1 - F2
    nrm = np.linalg.norm(d)
    dn = d / nrm if nrm > 1e-9 else None
    return F1, F2, dn


def selfcost(pts, r30, r45, a, t1, t2):
    """概率自洽代价 mean((probTotal - probFoci)²)"""
    pt, p1, p2, sigm = m.compute_probabilities_from_taus(r30, r45, t1, t2, a)
    F1, F2 = m.extract_foci(pts, p1, sigm[1], p2, sigm[2])
    pf = np.array([math.exp(-abs(m.dist(p, F1) + m.dist(p, F2) - 2.0 * a) / (2.0 * a))
                   for p in pts])
    return float(np.mean((pt - pf) ** 2)), F1, F2


def ellipcost(pts, F1, F2, a):
    return float(np.mean([(m.dist(p, F1) + m.dist(p, F2) - 2.0 * a) ** 2 for p in pts]))


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


# 之前批量测试中收敛出的 (t1f, t2f) —— 用作"固定 τ 普适性"候选
KNOWN_CONV = {
    "ball2":   ((0.0813, 1.1921), (-0.2402, 1.0161)),
    "ball3":   ((-0.2696, 1.5967), (0.0241, 1.8914)),
    "gauss0":  ((0.0377, 1.1118), (-0.0127, 1.4561)),
    "gauss1":  ((0.0133, 1.2313), (0.0330, 1.1049)),
    "gauss3":  ((0.2424, 1.8293), (0.1556, 1.0292)),
    "ellip0":  ((0.2307, 2.4625), (-0.2793, 1.9769)),
    "ellipN3": ((0.2402, 1.3853), (-0.3740, 1.0859)),
}


def main():
    sets = build_sets()
    prep = {k: prepare(v) for k, v in sets.items()}
    t10, t20 = m.compute_taus()
    x0 = np.array([t10.real, t10.imag, t20.real, t20.imag])
    xn = x0.copy()
    m.normalize_tau(xn)

    print("=" * 100)
    print("P1. 方向敏感性: 同一数据集, 不同 τ 的 d(τ) 与 PCA 轴 v3 的夹角")
    print("     (theory=理论初值  norm=归一化初值  ii=(i,i)  r1/r2/r3=基本域随机 τ)")
    print("=" * 100)
    cands = {
        "theory": (complex(x0[0], x0[1]), complex(x0[2], x0[3])),
        "norm":   (complex(xn[0], xn[1]), complex(xn[2], xn[3])),
        "ii":     (1j, 1j),
    }
    for name in ("ball0", "gauss0", "ellip0"):
        r30, r45, a, axis = prep[name]
        rng = np.random.default_rng(7)
        for k in range(3):
            cands[f"r{k}"] = (complex(rng.uniform(-0.5, 0.5), rng.uniform(0.8, 2.2)),
                              complex(rng.uniform(-0.5, 0.5), rng.uniform(0.8, 2.2)))
        angs = {}
        for lab, (t1, t2) in cands.items():
            _, _, dn = direction(sets[name], r30, r45, a, t1, t2)
            angs[lab] = ang_deg(dn, axis) if dn is not None else float("nan")
        print(f"{name:8s} axis={axis.round(3)}  " +
              "  ".join(f"{k}={v:.1f}°" for k, v in angs.items()))

    print()
    print("=" * 100)
    print("P2. 固定 τ 普适性: 用某集收敛的 (t1f,t2f) 套到其他所有集, 夹角是否都小?")
    print("     (若存在普适 τ, 某一列应全为小角)")
    print("=" * 100)
    names = list(sets.keys())
    header = "τ来源      " + "".join(f"{n:>8s}" for n in names)
    print(header)
    for src, ((r1, i1), (r2, i2)) in KNOWN_CONV.items():
        row = f"{src:<10s}"
        for name in names:
            r30, r45, a, axis = prep[name]
            _, _, dn = direction(sets[name], r30, r45, a, complex(r1, i1), complex(r2, i2))
            row += f"{ang_deg(dn, axis) if dn is not None else float('nan'):>8.1f}"
        print(row)

    print()
    print("=" * 100)
    print("P3. 去掉 PCA 的自洽环 (wDir=0): 能否自我收敛并靠近 v3?")
    print("=" * 100)
    for name in ("ball2", "gauss0", "ellip0"):
        r30, r45, a, axis = prep[name]
        # 理论τ 与 自洽环结果的对比
        c_th, F1t, F2t = selfcost(sets[name], r30, r45, a, t10, t20)
        dn_t = F1t - F2t
        dn_t = dn_t / np.linalg.norm(dn_t)
        t0 = time.time()
        t1f, t2f, F1s, F2s = m.refine_moduli_by_axis(sets[name], t10, t20, axis,
                                                     max_iter=40, w_dir=0.0,
                                                     verbose=False)
        dt = time.time() - t0
        c_self, _, _ = selfcost(sets[name], r30, r45, a, t1f, t2f)
        dns = F1s - F2s
        dns = dns / np.linalg.norm(dns)
        # PCA 收敛τ(若已知) 的自洽代价
        if name in KNOWN_CONV:
            (r1, i1), (r2, i2) = KNOWN_CONV[name]
            c_pca, F1p, F2p = selfcost(sets[name], r30, r45, a, complex(r1, i1), complex(r2, i2))
            dnp = F1p - F2p
            dnp = dnp / np.linalg.norm(dnp)
            print(f"{name:8s} 理论τ: 自洽={c_th:.5f} 角={ang_deg(dn_t, axis):.2f}° | "
                  f"自洽环: 自洽={c_self:.5f} 角={ang_deg(dns, axis):.2f}° ({dt:.0f}s) | "
                  f"PCA收敛τ: 自洽={c_pca:.5f} 角={ang_deg(dnp, axis):.2f}°")
        else:
            print(f"{name:8s} 理论τ: 自洽={c_th:.5f} 角={ang_deg(dn_t, axis):.2f}° | "
                  f"自洽环: 自洽={c_self:.5f} 角={ang_deg(dns, axis):.2f}° ({dt:.0f}s)")

    print()
    print("=" * 100)
    print("P4. 椭圆残差准则: 若仅用椭球残差 Σ(|p-F1|+|p-F2|-2a)^2 选 τ,")
    print("     最优 τ 对应的方向是否即 v3? (椭球数据应自洽)")
    print("=" * 100)
    for name in ("gauss0", "ellip0"):
        r30, r45, a, axis = prep[name]
        rng = np.random.default_rng(9)
        row = f"{name:8s}"
        for lab, (t1, t2) in list(cands.items()):
            F1, F2, dn = direction(sets[name], r30, r45, a, t1, t2)
            if dn is None:
                continue
            ang = ang_deg(dn, axis)
            ec = ellipcost(sets[name], F1, F2, a)
            row += f" | {lab}: 残差={ec:.4f} 角={ang:.1f}°"
        print(row)

    print()
    print("DONE")


if __name__ == "__main__":
    main()
