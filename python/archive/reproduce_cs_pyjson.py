# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
精确复现 C# n2sjy2 (pyjson): angleDegodr 机制 + NM
  odr = RefineModuliByAxis 内层首次 ∠(F1-F2, axis)  [外环 reset 后内层第一轮捕获]
  new = 外环开始时 ∠(f1-f2, f01-f02)
  测试: a9 能否缩到 2? 终止条件 |odr-new|>16 (和 &&a9<1) 何时触发?
"""
import json
import math
import numpy as np

import data_driven_axis as dd
import n2sjy2 as n2
import nsjy_algorithms as m
from scipy.optimize import minimize

PYJSON = r"C:\Users\23128\My project (2)\Assets\Resources\pyjson.json"


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


def refine_lm(pts, r30, r45, a, t10, t20, axis_v, n_iter=50, capture_odr=True,
              odr_state=0.0):
    """RefineModuliByAxis: 返回 (t1,t2,F1,F2, 捕获的odr)"""
    P = np.array(pts, float)
    axis = np.asarray(axis_v, float)
    axis = axis / np.linalg.norm(axis)
    x = np.array([t10.real, t10.imag, t20.real, t20.imag], float)
    captured = None

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
            # C#: if(angleDegodr==0) angleDegodr=angleDeg   (外环已reset为0)
            if capture_odr and odr_state == 0.0 and captured is None:
                captured = ang
        if ang < 0.5:
            break
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
    return complex(x[0], x[1]), complex(x[2], x[3]), F1, F2, captured


def rotate(vv, ax, deg):
    ax = np.asarray(ax, float)
    if np.linalg.norm(ax) < 1e-12:
        return vv.copy()
    ax = ax / np.linalg.norm(ax)
    rg = math.radians(deg)
    v = vv * math.cos(rg) + np.cross(ax, vv) * math.sin(rg) \
        + ax * np.dot(ax, vv) * (1 - math.cos(rg))
    return v / np.linalg.norm(v)


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
        return obj(xk) <= 16.

    try:
        rr = minimize(obj, x0, method="Nelder-Mead", bounds=list(zip(lb, ub)),
                      callback=cb, options={"maxiter": max_evals, "maxfev": max_evals * 2,
                                            "xatol": 1e-6, "fatol": 1e-6})
        best = np.clip(rr.x, lb, ub)
    except Exception:
        best = x0
    obj(best)
    return complex(best[0], best[1]), complex(best[2], best[3])


def run(max_outer=60, use_cond2=False):
    with open(PYJSON, "r", encoding="utf-8") as f:
        raw = json.load(f)
    pts = [np.array([p["x"], p["y"], p["z"]], float) for p in raw]
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
    v3pca = m.pca(pts)[1]
    v3pca = v3pca / np.linalg.norm(v3pca)
    C = P.T @ P
    evals, evecs = np.linalg.eigh(C)
    v_true = evecs[:, np.argsort(evals)[::-1][0]]

    def norm(vv):
        return np.asarray(vv, float) / np.linalg.norm(vv)

    v2 = norm(f1 - f2)
    v4 = norm(f01 - f02)
    f01fit = norm(f01 - f02)
    angleDegodr = 0.0   # 外环实例字段, reset at 152
    t1, t2 = t_, t_
    f1z, f2z = f1, f2
    best_a9v, best_a9t = 999.0, 999.0
    print(f"精确复现 (pyjson), 终止 |odr-new|>16 {'&& a9<1' if use_cond2 else ''}:")
    print(f"  {'iter':>4s} {'a9(v3)':>7s} {'a9(真)':>7s} {'odr':>6s} {'new':>6s} {'|odr-new|':>9s} {'stop?':>6s}")
    stop = None
    for i in range(max_outer):
        d_cur = norm(f1z - f2z)
        # angleDegnew = ∠(f1-f2, f01-f02)  (C# 第90行: 用拟合)
        angNew = angle(f01fit, d_cur)
        angleDegodr = 0.0   # C# 152: 每轮 reset
        # RefineModuliByAxis(axis = d 自指)
        t1n, t2n, F1n, F2n, captured = refine_lm(
            pts, r30, r45, a, t_, t_, d_cur, n_iter=50,
            capture_odr=True, odr_state=angleDegodr)
        odr = captured if captured is not None else 0.0
        angleDegodr = odr
        ptn, p1n, p2n, sgn = dd.fast_probs(r30, r45, t1n, t2n, a)
        f1n, f2n = fit_fast(P, ptn, F1n, F2n, a)
        v6 = norm(f1n - f2n)
        a9v = angle(v6, v3pca)
        a9t = angle(v6, v_true)
        best_a9v = min(best_a9v, a9v)
        best_a9t = min(best_a9t, a9t)
        diff = abs(odr - angNew)
        cond_ok = diff > 16
        cond_ok2 = cond_ok and a9v < 1
        stop_flag = cond_ok2 if use_cond2 else cond_ok
        if i % 3 == 0 or stop_flag:
            print(f"  {i:4d} {a9v:7.2f} {a9t:7.2f} {odr:6.1f} {angNew:6.1f} "
                  f"{diff:9.1f} {str(stop_flag):>6s}", flush=True)
        if stop_flag:
            stop = i
            break
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
    print(f"  ==> stop at iter{stop} | best a9(v3)={best_a9v:.2f} a9(真)={best_a9t:.2f}")
    return best_a9v, best_a9t


def main():
    print("=" * 90)
    av, at = run(max_outer=60, use_cond2=False)
    print()
    av2, at2 = run(max_outer=60, use_cond2=True)
    print("DONE")


if __name__ == "__main__":
    main()
