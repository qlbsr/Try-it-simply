# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
全数据集测试用户最新策略 (n2sjy2.cs 第71-87行):
  angleDegnew = ∠(F01-F02, f1-f2)
  if angleDegnew < 60:  // 活性不足
      axi = 与(F01-F02)成60° 且绕其随机方位 的方向
  RefineModuliByAxis(..., axi, 50)   // 用扰动的 axi (而非 d 自指)
统计: a9(v3)=∠(v6,pca的v3) 与 a9(真)=∠(v6,真主轴) 的最小值 (多 seeds best)
"""
import math
import numpy as np

import data_driven_axis as dd
import n2sjy2 as n2
import nsjy_algorithms as m
from scipy.optimize import minimize


def angle(u, vv):
    return math.degrees(math.acos(np.clip(np.dot(u, vv) / (np.linalg.norm(u) * np.linalg.norm(vv)), -1, 1)))


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


def rotate(vv, ax, deg):
    ax = np.asarray(ax, float)
    if np.linalg.norm(ax) < 1e-12:
        return vv.copy()
    ax = ax / np.linalg.norm(ax)
    rg = math.radians(deg)
    v = vv * math.cos(rg) + np.cross(ax, vv) * math.sin(rg) \
        + ax * np.dot(ax, vv) * (1 - math.cos(rg))
    return v / np.linalg.norm(v)


def refine_lm(pts, r30, r45, a, t10, t20, axis_v, n_iter=50):
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
        r[len(pts)] = 1.0 * e[0]
        r[len(pts) + 1] = 1.0 * e[1]
        r[len(pts) + 2] = 1.0 * e[2]
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


def nm_refine(pts, r30, r45, a, t1, t2, ref1, ref2, max_evals=100):
    P = np.array(pts, float)
    lb = np.array([-0.5, 0.5, -0.5, 0.5])
    ub = np.array([0.5, 2.0, 0.5, 2.0])

    def obj(x):
        tt1 = complex(x[0], x[1])
        tt2 = complex(x[2], x[3])
        ptc, pc1, pc2, sc = dd.fast_probs(r30, r45, tt1, tt2, a)
        Fc1, Fc2 = m.extract_foci(pts, pc1, sc[1], pc2, sc[2])
        fc1, fc2 = fit_fast(P, ptc, Fc1, Fc2, a)
        dd_ = fc1 - fc2
        if np.dot(dd_, dd_) < 1e-12:
            return 0.
        dd_ = dd_ / np.linalg.norm(dd_)
        return abs(angle(dd_, ref1) - angle(dd_, ref2))

    x0 = np.clip([t1.real, t1.imag, t2.real, t2.imag], lb, ub)

    def cb(xk):
        return obj(xk) <= 0.5

    try:
        rr = minimize(obj, x0, method="Nelder-Mead", bounds=list(zip(lb, ub)),
                      callback=cb, options={"maxiter": max_evals, "maxfev": max_evals * 2,
                                            "xatol": 1e-6, "fatol": 1e-6})
        best = np.clip(rr.x, lb, ub)
    except Exception:
        best = x0
    obj(best)
    return complex(best[0], best[1]), complex(best[2], best[3])


def run_user_strategy(name, pts, seed=0, n_rounds=60, perturb=True):
    """忠实用户逻辑: angleDegnew<60 → axi=60°随机方位; LM用axi"""
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
    pt, p1, p2, sig = dd.fast_probs(r30, r45, t1t, t2t, a)
    F1, F2 = m.extract_foci(pts, p1, sig[1], p2, sig[2])
    f1, f2 = fit_fast(P, pt, F1, F2, a)
    pt01, p101, p201, sig01 = dd.fast_probs(r30, r45, t_, t_, a)
    F01, F02 = m.extract_foci(pts, p101, sig01[1], p201, sig01[2])
    f01, f02 = fit_fast(P, pt01, F01, F02, a)
    v3 = m.pca(pts)[1]
    v3 = v3 / np.linalg.norm(v3)
    C = P.T @ P
    evals, evecs = np.linalg.eigh(C)
    v_true = evecs[:, np.argsort(evals)[::-1][0]]

    def norm(vv):
        return np.asarray(vv, float) / np.linalg.norm(vv)

    v2 = norm(f1 - f2)
    v4 = norm(f01 - f02)
    F01d = norm(F01 - F02)
    f01d_fit = norm(f01 - f02)
    rng = np.random.default_rng(seed)
    t1, t2 = t_, t_
    f1z, f2z = f1, f2
    a9_best, a9t_best = 999.0, 999.0
    n_pert = 0
    for i in range(n_rounds):
        d_cur = norm(f1z - f2z)
        # angleDegnew = ∠(f01-f02, f1-f2)  (用户第67行: 用拟合的 f01-f02)
        angNew = angle(f01d_fit, d_cur)
        axi = d_cur
        if perturb and angNew < 60:
            n_pert += 1
            r = F01d  # 用户第69行用 (F01-F02)
            n_ = np.cross(d_cur, r)
            if np.linalg.norm(n_) < 1e-4:
                tmp = np.array([1., 0, 0]) if abs(d_cur[0]) < 0.9 else np.array([0., 1, 0])
                n_ = np.cross(d_cur, tmp)
                n_ = n_ / np.linalg.norm(n_)
            dPerp = d_cur - np.dot(d_cur, r) * r
            if np.linalg.norm(dPerp) > 1e-6:
                dPerp = dPerp / np.linalg.norm(dPerp)
                onCone = 0.5 * r + 0.866 * dPerp
                az = rng.uniform(0, 2 * math.pi)
                axi = rotate(onCone, r, math.degrees(az))
            else:
                axi = d_cur
        # LM (用户第88行: 若接上 axi)
        t1n, t2n, F1n, F2n = refine_lm(pts, r30, r45, a, t_, t_, axi)
        ptn, p1n, p2n, sgn = dd.fast_probs(r30, r45, t1n, t2n, a)
        f1n, f2n = fit_fast(P, ptn, F1n, F2n, a)
        v6 = norm(f1n - f2n)
        a9 = angle(v6, v3)
        a9t = angle(v6, v_true)
        a9_best = min(a9_best, a9)
        a9t_best = min(a9t_best, a9t)
        ang5 = angle(v6, v2)
        ang6 = angle(v6, v4)
        axis = np.cross(v6, v4)
        axis1 = np.cross(v6, v2)
        newf = rotate(v4, axis, -ang5)
        newv = rotate(v2, axis1, -ang6)
        t1, t2 = nm_refine(pts, r30, r45, a, t1n, t2n, newf, newv)
        ptn, p1n, p2n, sgn = dd.fast_probs(r30, r45, t1, t2, a)
        F1n, F2n = m.extract_foci(pts, p1n, sgn[1], p2n, sgn[2])
        f1z, f2z = fit_fast(P, ptn, F1n, F2n, a)
    return a9_best, a9t_best, n_pert


def main():
    import json
    dats = {}
    with open(r"C:\Users\23128\My project (2)\Assets\Resources\pyjson.json", encoding="utf-8") as f:
        raw = json.load(f)
    dats["pyjson"] = [np.array([p["x"], p["y"], p["z"]], float) for p in raw]
    for seed in range(4):
        dats[f"ball{seed}"] = dd.make_ball(seed)
    for seed in range(4):
        dats[f"ellip{seed}"] = dd.make_ellip(seed)
    for seed in range(3):
        dats[f"cube{seed}"] = dd.make_cube(10 + seed)
    # gauss
    dats["gauss5"] = [np.random.default_rng(1005).standard_normal(3) * (0.5 + 2 * np.random.default_rng(1006).random()) for _ in range(200)]

    print("=" * 110)
    print("全数据集 (200点): 用户 60°随机方位策略 | 3 seeds 取 best")
    print("=" * 110)
    print(f"{'数据集':10s} {'a9(v3)best':>10s} {'a9(真)best':>10s} {'≤10°(v3)':>9s} {'≤10°(真)':>9s}")
    res = []
    for name, pts in dats.items():
        best_v3, best_t = 999.0, 999.0
        try:
            for sd in range(3):
                av, at, np_ = run_user_strategy(name, pts, seed=sd)
                best_v3 = min(best_v3, av)
                best_t = min(best_t, at)
            res.append((name, best_v3, best_t))
            print(f"{name:10s} {best_v3:10.1f} {best_t:10.1f} "
                  f"{'✅' if best_v3<=10 else '✗':>9s} {'✅' if best_t<=10 else '✗':>9s}",
                  flush=True)
        except Exception as e:
            print(f"{name:10s} 失败 {e}", flush=True)
    print()
    n10v3 = sum(1 for _, a, _ in res if a <= 10)
    n10t = sum(1 for _, _, b in res if b <= 10)
    n30t = sum(1 for _, _, b in res if b <= 30)
    print(f"a9(v3)≤10°: {n10v3}/{len(res)} | a9(真)≤10°: {n10t}/{len(res)} | a9(真)≤30°: {n30t}/{len(res)}")
    print("DONE")


if __name__ == "__main__":
    main()
