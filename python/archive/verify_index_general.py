# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
验证: 规律 s0≈s5 与 probs12z 收敛是只在 [0] 成立, 还是对所有点 [i] 成立?
  对达成数据集 (ellip0/ball0), 在达成点检查:
    s0[i] vs s5[i]: 全数组差异 (均值/最大/占比 |s0-s5|*100 < 2?)
    probs12[0][i] vs probs012[0][i] vs probs12z[0][i]: 全点对比
  注意 s1..s5 是 BatchProbability(焦点对) 逐点数组 (200 元素)
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


def analyze(name, pts, max_outer=12):
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
            # 达成点: 检查全数组
            s0 = bp(v * c, -v * c)
            s5 = bp(d * c, -d * c)
            s2 = bp(v1 * c, -v1 * c)   # v1 方向焦点对
            s4 = bp(v3 * c, -v3 * c)   # v3 方向焦点对
            pt_cur, _, _, _ = dd.fast_probs(r30, r45, t1, t2, a)
            print(f"  [{name}] 达成 iter{i} cond={cond:.2f}")
            # s0 vs s5 全数组
            d05 = np.abs(s0 - s5) * 100
            d25 = np.abs(s2 - s5) * 100
            d45 = np.abs(s4 - s5) * 100
            print(f"    s0-s5 (×100): [0]={d05[0]:.2f} [1]={d05[1]:.2f} [2]={d05[2]:.2f} "
                  f"中位={np.median(d05):.2f} P90={np.percentile(d05,90):.2f} max={d05.max():.2f}")
            print(f"    s2-s5 (×100): [0]={d25[0]:.2f} [1]={d25[1]:.2f} 中位={np.median(d25):.2f}")
            print(f"    s4-s5 (×100): [0]={d45[0]:.2f} [1]={d45[1]:.2f} 中位={np.median(d45):.2f}")
            # 全数组 s5 是否最小?
            mins = np.argmin(np.vstack([s0, s2, s4, s5]), axis=0)
            frac_min = (mins == 3).mean() * 100
            print(f"    s5 为最小值的点占比: {frac_min:.0f}%")
            # probs12/012/12z 全数组
            d_z012 = np.abs(pt_cur - pt01) * 100
            d_z12 = np.abs(pt_cur - pt) * 100
            print(f"    |probs12z-probs012| (×100): [0]={d_z012[0]:.2f} [1]={d_z012[1]:.2f} "
                  f"中位={np.median(d_z012):.2f} P90={np.percentile(d_z012,90):.2f}")
            print(f"    |probs12z-probs12 | (×100): [0]={d_z12[0]:.2f} [1]={d_z12[1]:.2f} "
                  f"中位={np.median(d_z12):.2f}")
            return
        if i >= max_outer - 1:
            print(f"  [{name}] 未达成 (cond={cond:.1f})")
            return
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


def main():
    print("=" * 96)
    print("[0] vs [i]: 规律是否对所有点成立?")
    print("=" * 96)
    for name, pts in [("ellip0", dd.make_ellip(0)), ("ball0", dd.make_ball(0)),
                      ("ball2", dd.make_ball(2)), ("gauss5",
                      [np.random.default_rng(1005).standard_normal(3) * (0.5 + 2 * np.random.default_rng(1006).random()) for _ in range(200)])]:
        try:
            analyze(name, pts)
        except Exception as e:
            print(f"  {name}: 失败 ({e})")
        print(flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
