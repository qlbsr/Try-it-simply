# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
尝试停止条件: |angleDegodr - angleDegnew| 最大时(档位跳变峰值) 且 angleDeg9<1
  记录全程 odr/new/|Δ|/a9, 找 |Δ| 局部峰值, 检查峰值处 a9 是否<1
  注意: angleDeg9=∠(v6, v3), v3=pca()重建(与真主轴差~40°), a9<1 极难
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


def refine_lm_track(pts, r30, r45, a, t10, t20, axis_v, n_iter=50):
    """RefineModuliByAxis: 返回 (t1,t2,F1,F2, 首次angleDeg=odr)"""
    P = np.array(pts, float)
    axis = np.asarray(axis_v, float)
    axis = axis / np.linalg.norm(axis)
    x = np.array([t10.real, t10.imag, t20.real, t20.imag], float)
    first_angle = None

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
        dv = F1 - F2
        ang = 180.0
        if np.linalg.norm(dv) > 1e-12:
            dv = dv / np.linalg.norm(dv)
            if np.dot(dv, axis) < 0:
                dv = -dv
            ang = angle(dv, axis)
            if first_angle is None:
                first_angle = ang
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
    return complex(x[0], x[1]), complex(x[2], x[3]), F1, F2, first_angle


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


def rotate(vv, a_, deg):
    if np.linalg.norm(a_) < 1e-12:
        return vv
    a_ = a_ / np.linalg.norm(a_)
    rg = math.radians(deg)
    return vv * math.cos(rg) + np.cross(a_, vv) * math.sin(rg) \
        + a_ * np.dot(a_, vv) * (1 - math.cos(rg))


def run_peak(name, pts, n_rounds=60):
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
    v3 = v3 / np.linalg.norm(v3)     # 用户 v = pca() 的 v3
    C = P.T @ P
    evals, evecs = np.linalg.eigh(C)
    v_true = evecs[:, np.argsort(evals)[::-1][0]]

    def norm(vv):
        return np.asarray(vv, float) / np.linalg.norm(vv)

    v2 = norm(f1 - f2)
    v4 = norm(f01 - f02)
    F01d = norm(F01 - F02)
    t1, t2 = t_, t_
    f1z, f2z = f1, f2
    rows = []
    for i in range(n_rounds):
        d_cur = norm(f1z - f2z)
        angNew = angle(F01d, d_cur)          # angleDegnew
        t1n, t2n, F1n, F2n, first_ang = refine_lm_track(
            pts, r30, r45, a, t_, t_, d_cur)
        odr = first_ang if first_ang is not None else angNew
        ptn, p1n, p2n, sgn = dd.fast_probs(r30, r45, t1n, t2n, a)
        f1n, f2n = fit_fast(P, ptn, F1n, F2n, a)
        v6 = norm(f1n - f2n)
        a9_v3 = angle(v6, v3)                # angleDeg9 (对用户v3)
        a9_true = angle(v6, v_true)          # 对真主轴 (测量)
        rows.append((odr, angNew, abs(odr - angNew), a9_v3, a9_true, i, t1n, t2n))
        ang5 = angle(v6, v2)
        ang6 = angle(v6, v4)
        axis = np.cross(v6, v2)
        axis1 = np.cross(v6, v4)
        newf = rotate(v2, axis, -ang5)
        newv = rotate(v4, axis1, -ang6)
        t1, t2 = nm_refine(pts, r30, r45, a, t1n, t2n, newf, newv)
        ptn, p1n, p2n, sgn = dd.fast_probs(r30, r45, t1, t2, a)
        F1n, F2n = m.extract_foci(pts, p1n, sgn[1], p2n, sgn[2])
        f1z, f2z = fit_fast(P, ptn, F1n, F2n, a)
    # 分析: |Δ| 峰值 (局部最大)
    print(f"  [{name}]")
    print(f"    {'iter':>4s} {'odr':>6s} {'new':>6s} {'|Δ|':>6s} {'a9(v3)':>7s} {'a9(真)':>7s}")
    for k in range(len(rows)):
        odr, angNew, dlt, a9v, a9t, i, t1n, t2n = rows[k]
        if k % 4 == 0:
            print(f"    {i:4d} {odr:6.1f} {angNew:6.1f} {dlt:6.1f} {a9v:7.2f} {a9t:7.2f}")
    # 找 |Δ| 全局最大与 a9 最小
    dmax = max(rows, key=lambda r: r[2])
    a9min = min(rows, key=lambda r: r[3])
    a9tmin = min(rows, key=lambda r: r[4])
    print(f"    |Δ|最大: iter{dmax[5]} odr={dmax[0]:.1f} new={dmax[1]:.1f} "
          f"|Δ|={dmax[2]:.1f} a9(v3)={dmax[3]:.2f} a9(真)={dmax[4]:.2f}")
    print(f"    a9(v3)最小: iter{a9min[5]} = {a9min[3]:.2f}° (|Δ|={a9min[2]:.1f})")
    print(f"    a9(真)最小: iter{a9tmin[5]} = {a9tmin[4]:.2f}° (|Δ|={a9tmin[2]:.1f} odr={a9tmin[0]:.1f} new={a9tmin[1]:.1f})")
    # 停止条件模拟: |Δ|>9 且 a9<1?
    hits = [r for r in rows if r[2] > 9 and r[3] < 1]
    print(f"    满足 |Δ|>9 且 a9(v3)<1: {len(hits)} 轮")
    print()


def main():
    import json
    print("=" * 100)
    print("停止条件尝试: |odr-new| 峰值 + a9<1")
    print("=" * 100)
    dats = {}
    with open(r"C:\Users\23128\My project (2)\Assets\Resources\pyjson.json", encoding="utf-8") as f:
        raw = json.load(f)
    dats["pyjson"] = [np.array([p["x"], p["y"], p["z"]], float) for p in raw]
    dats["ball2"] = dd.make_ball(2)
    dats["ellip0"] = dd.make_ellip(0)
    for nm, pts in dats.items():
        try:
            run_peak(nm, pts)
        except Exception as e:
            print(f"  {nm}: 失败 ({e})", flush=True)
        print(flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
