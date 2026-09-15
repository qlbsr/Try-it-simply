# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
[1][2] 等索引的"普遍性"是否与 [0] 一样?
  前提澄清: 判据#2 cond<=8 是方向角(全局量), 77% 是数据集达成率, 与索引无关.
  本实验: 在多个达成数据集上, 检查达成点各索引 [i] 的 |s0[i]-s5[i]|×100:
    - 若 s0≈s5 是"达成特征", 则所有索引应普遍小 (高达成率)
    - 若只是 [0] 局部, 则各索引差异大
  统计: 每个索引 i 在多少数据集中 |s0-s5|*100 < 5 (逐索引达成率)
"""
import math
import numpy as np

import data_driven_axis as dd
import n2sjy2 as n2
import nsjy_algorithms as m
from scipy.optimize import minimize


def angle(u, v):
    return math.degrees(math.acos(np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1)))


def fit_fast(P, prob, F1, F2, a, iters=300, lr=0.0001):
    F1 = np.array(F1, float).copy()
    F2 = np.array(F2, float).copy()
    for _ in range(iters):
        d1 = np.linalg.norm(P - F1, axis=1)
        d2 = np.linalg.norm(P - F2, axis=1)
        sd1 = np.maximum(d1, 1e-3)
        sd2 = np.maximum(d2, 1e-3)
        delta = d1 + d2 - 2 * a
        fp = np.exp(-np.abs(delta) / (2 * a))
        diff = fp - prob
        loss = float(np.sum(diff * diff))
        if loss < 1e-12:
            break
        sign = np.where(delta >= 0, 1., -1.)
        coef = diff * fp * sign / (2 * a)
        g1 = coef[:, None] * (P - F1) / sd1[:, None]
        g2 = coef[:, None] * (P - F2) / sd2[:, None]
        grad1 = g1.sum(0)
        grad2 = g2.sum(0)
        sep = F1 - F2
        grad1 += 2 * sep
        grad2 -= 2 * sep
        n1 = np.linalg.norm(grad1)
        n2 = np.linalg.norm(grad2)
        if n1 > 5:
            grad1 *= 5 / n1
        if n2 > 5:
            grad2 *= 5 / n2
        F1 -= lr * grad1
        F2 -= lr * grad2
        if not (np.all(np.isfinite(F1)) and np.all(np.isfinite(F2))):
            break
    return F1, F2


def nm_delta(t1i, t2i, pts, r30, r45, a, pcav, f1f2v, max_evals=400):
    lb = np.array([-0.5, 0.5, -0.5, 0.5])
    ub = np.array([0.5, 1.5, 0.5, 1.5])
    last = {}
    P = np.array(pts, float)

    def obj(x):
        t1 = complex(x[0], x[1])
        t2 = complex(x[2], x[3])
        pt, p1, p2, sig = dd.fast_probs(r30, r45, t1, t2, a)
        F1, F2 = m.extract_foci(pts, p1, sig[1], p2, sig[2])
        f1, f2 = fit_fast(P, pt, F1, F2, a)
        d = f1 - f2
        last["F1z"], last["F2z"] = f1, f2
        if np.dot(d, d) < 1e-12:
            return 0.
        d = d / np.linalg.norm(d)
        ad = angle(d, pcav)
        ad1 = angle(d, f1f2v)
        last["ad"], last["ad1"] = ad, ad1
        return abs(ad - ad1)

    x0 = np.clip([t1i.real, t1i.imag, t2i.real, t2i.imag], lb, ub)

    def cb(xk):
        return obj(xk) <= 10.

    res = minimize(obj, x0, method="Nelder-Mead", bounds=list(zip(lb, ub)),
                   callback=cb, options={"maxiter": max_evals, "maxfev": max_evals * 2,
                                         "xatol": 1e-6, "fatol": 1e-6})
    best = np.clip(res.x, lb, ub)
    obj(best)
    return (complex(best[0], best[1]), complex(best[2], best[3]),
            last.get("F1z"), last.get("F2z"))


def refine_lm(pts, r30, r45, a, t10, t20, axis_v, n_iter=8):
    P = np.array(pts, float)
    axis = np.asarray(axis_v, float)
    axis = axis / np.linalg.norm(axis)
    x = np.array([t10.real, t10.imag, t20.real, t20.imag], float)

    def resid(xx):
        t1 = complex(xx[0], xx[1])
        t2 = complex(xx[2], xx[3])
        pt, p1, p2, sig = dd.fast_probs(r30, r45, t1, t2, a)
        F1c, F2c = m.extract_foci(pts, p1, sig[1], p2, sig[2])
        r = np.zeros(len(pts) + 9)
        d1 = np.linalg.norm(P - F1c, axis=1)
        d2 = np.linalg.norm(P - F2c, axis=1)
        delta = d1 + d2 - 2 * a
        pf = np.exp(-np.abs(delta) / (2 * a))
        r[:len(pts)] = (pt - pf) / math.sqrt(len(pts))
        dv = F1c - F2c
        if np.linalg.norm(dv) < 1e-12:
            dv = axis
        dv = dv / np.linalg.norm(dv)
        if np.dot(dv, axis) < 0:
            dv = -dv
        e = dv - axis
        r[len(pts)] = 20 * e[0]
        r[len(pts) + 1] = 20 * e[1]
        r[len(pts) + 2] = 20 * e[2]
        return r, F1c, F2c

    lam = 1e-3
    for it in range(n_iter):
        r, F1, F2 = resid(x)
        cost = float(np.sum(r * r))
        J = np.zeros((len(r), 4))
        for k in range(4):
            xp = x.copy()
            xp[k] += 1e-3
            rp2, _, _ = resid(xp)
            J[:, k] = (rp2 - r) / 1e-3
        A = J.T @ J
        g = J.T @ r
        Aaug = A.copy()
        for c in range(4):
            Aaug[c, c] += lam * (A[c, c] + 1e-12)
        try:
            delta = np.linalg.solve(Aaug, g)
        except Exception:
            break
        xtry = x - delta
        r2, _, _ = resid(xtry)
        cost2 = float(np.sum(r2 * r2))
        if cost2 < cost:
            x = xtry
            lam = max(lam / 3, 1e-8)
        else:
            lam = min(lam * 3, 1e4)
    r, F1, F2 = resid(x)
    return complex(x[0], x[1]), complex(x[2], x[3]), F1, F2


def run_until_achieve(name, pts, max_outer=15):
    """跑到达成, 返回达成点全部 s0/s5 数组; 未达成返回 None"""
    pts = [np.array(p, float) for p in pts]
    P = np.array(pts, float)
    rp = n2.compute_rp(pts)
    d2 = math.asin(math.cos(math.radians(30)) / math.pi)
    a = rp * (1 + math.sin(d2))
    e3 = math.cos(d2) * 2 / (1 + math.sin(d2))
    c = rp * math.cos(d2) * e3
    _, r45, r30, _ = n2.yzqx(pts, rp)
    t1t, t2t = n2.compute_taus()
    t_ = complex(0, 1)

    def bp(F1, F2):
        d1 = np.linalg.norm(P - F1, axis=1)
        d2 = np.linalg.norm(P - F2, axis=1)
        delta = d1 + d2 - 2 * a
        return np.exp(-np.abs(delta) / (2 * a))

    pt, p1, p2, sig = dd.fast_probs(r30, r45, t1t, t2t, a)
    F1, F2 = m.extract_foci(pts, p1, sig[1], p2, sig[2])
    f1, f2 = fit_fast(P, pt, F1, F2, a)
    pt01, p101, p201, sig01 = dd.fast_probs(r30, r45, t_, t_, a)
    F01, F02 = m.extract_foci(pts, p101, sig01[1], p201, sig01[2])
    f01, f02 = fit_fast(P, pt01, F01, F02, a)
    v = m.pca(pts)[1]
    v = v / np.linalg.norm(v)

    def norm(vv):
        return np.asarray(vv, float) / np.linalg.norm(vv)

    v1 = norm(F1 - F2)
    v3 = norm(F01 - F02)
    vj = v + norm(F1 - F2)
    refine_lm(pts, r30, r45, a, t_, t_, vj)
    t1, t2 = t_, t_
    f1z, f2z = f1, f2
    for i in range(max_outer):
        d = norm(f1z - f2z)
        ad1 = angle(d, v1)
        ad2 = angle(d, v3)
        cond = abs((ad2 + ad1) * 0.5 - min(ad2, ad1))
        if cond <= 8:
            s0 = bp(v * c, -v * c)
            s5 = bp(d * c, -d * c)
            return s0, s5, i, cond
        if i >= max_outer - 1:
            return None
        t1n, t2n, F1n, F2n = refine_lm(pts, r30, r45, a, t_, t_, norm(f1z - f2z))
        axis = np.cross(norm(F1n - F2n), norm(F1 - F2))
        axis1 = np.cross(norm(F1n - F2n), norm(F01 - F02))

        def rotate(vv, ax, deg):
            if np.linalg.norm(ax) < 1e-12:
                return vv
            ax = ax / np.linalg.norm(ax)
            rg = math.radians(deg)
            return vv * math.cos(rg) + np.cross(ax, vv) * math.sin(rg) \
                + ax * np.dot(ax, vv) * (1 - math.cos(rg))

        newv = rotate(norm(F01 - F02), axis, -ad2)
        newF = rotate(norm(F1 - F2), axis1, -ad2)
        t1, t2, F1z, F2z = nm_delta(t1n, t2n, pts, r30, r45, a, newv, newF)
        f1z, f2z = F1z, F2z
    return None


def main():
    datasets = {}
    for seed in range(4):
        datasets[f"ellip{seed}"] = dd.make_ellip(seed)
    for seed in range(4):
        datasets[f"ball{seed}"] = dd.make_ball(seed)
    for seed in range(3):
        datasets[f"cube{seed}"] = dd.make_cube(10 + seed)

    print("=" * 96)
    print("达成点各索引 |s0[i]-s5[i]|×100: [0] 是否特殊?")
    print("=" * 96)
    # 收集所有达成数据集的 d05 数组
    d05_all = []   # list of arrays
    names = []
    for name, pts in datasets.items():
        try:
            res = run_until_achieve(name, pts)
        except Exception as e:
            print(f"  {name}: 失败 ({e})", flush=True)
            continue
        if res is None:
            print(f"  {name}: 未达成 (跳过)", flush=True)
            continue
        s0, s5, it, cond = res
        d05 = np.abs(s0 - s5) * 100
        d05_all.append(d05)
        names.append(name)
        print(f"  {name:8s}: 达成 iter{it} cond={cond:.2f} | d05[0]={d05[0]:5.2f} "
              f"d05[1]={d05[1]:5.2f} d05[2]={d05[2]:5.2f} d05[3]={d05[3]:5.2f} "
              f"中位={np.median(d05):5.2f} P90={np.percentile(d05,90):5.2f}", flush=True)

    print()
    if len(d05_all) < 3:
        print("达成数据集不足, 无法统计")
        return
    D = np.vstack(d05_all)
    print("=" * 96)
    print("逐索引达成率: |s0[i]-s5[i]|×100 < 5 的数据集占比 (共 %d 个达成数据集)" % D.shape[0])
    print("=" * 96)
    for i in [0, 1, 2, 3, 5, 10, 20, 50, 100, 150, 199]:
        frac = (D[:, i] < 5).mean() * 100
        print(f"  index [{i:3d}]: |s0-s5|<5 占比 {frac:5.1f}%   "
              f"均值 {D[:, i].mean():5.2f}  中位 {np.median(D[:, i]):5.2f}")
    # 全体点平均
    allfrac = (D < 5).mean() * 100
    print(f"  全部点: |s0-s5|<5 占比 {allfrac:.1f}%")
    print()
    print("若各索引占比≈[0] → s0≈s5 是普遍达成特征; 若 [0] 显著不同 → 局部现象")
    print("DONE")


if __name__ == "__main__":
    main()
