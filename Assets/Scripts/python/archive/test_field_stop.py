# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
真实迭代测试: 场条件 vs 角度条件 的触发对比 (诚实评估)
  角度条件: (angleDeg+angleDeg1)/2 - min <= 8   [|Δ|≤16]
  场条件候选: |corr(s5,s1)-corr(s5,s3)| < T    [无 PCA]
  在完整迭代流程中同时记录两者, 看触发轮次/时刻是否一致
"""
import json
import math
import time

import numpy as np
from scipy.optimize import minimize

import data_driven_axis as dd
import n2sjy2 as n2
import nsjy_algorithms as m

PYJSON = r"C:\Users\23128\My project (2)\Assets\Resources\pyjson.json"


def angle(u, v):
    un = np.linalg.norm(u)
    vn = np.linalg.norm(v)
    if un < 1e-12 or vn < 1e-12:
        return 0.0
    return math.degrees(math.acos(np.clip(np.dot(u, v) / (un * vn), -1, 1)))


def batch_probability(points, F1, F2, a):
    P = np.asarray(points, float)
    d1 = np.linalg.norm(P - np.asarray(F1, float), axis=1)
    d2 = np.linalg.norm(P - np.asarray(F2, float), axis=1)
    delta = d1 + d2 - 2.0 * a
    return np.exp(-np.abs(delta) / (2.0 * a))


def corr(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def fit_foci_fast(points, true_prob, init_f1, init_f2, a, max_iter=300, lr=0.0001, lambda_sep=1.0):
    P = np.asarray(points, float)
    prob = np.asarray(true_prob, float)
    F1 = np.array(init_f1, float)
    F2 = np.array(init_f2, float)
    eps = 1e-3
    for _ in range(max_iter):
        d1 = np.linalg.norm(P - F1, axis=1)
        d2 = np.linalg.norm(P - F2, axis=1)
        sd1 = np.maximum(d1, eps)
        sd2 = np.maximum(d2, eps)
        delta = d1 + d2 - 2.0 * a
        fit_prob = np.exp(-np.abs(delta) / (2.0 * a))
        diff = fit_prob - prob
        loss = float(np.sum(diff * diff))
        if loss < 1e-12:
            break
        sign = np.where(delta >= 0, 1.0, -1.0)
        coef = diff * fit_prob * sign / (2.0 * a)
        g1 = coef[:, None] * (P - F1) / sd1[:, None]
        g2 = coef[:, None] * (P - F2) / sd2[:, None]
        n1 = np.linalg.norm(g1, axis=1, keepdims=True)
        n2 = np.linalg.norm(g2, axis=1, keepdims=True)
        g1 = np.where(n1 > 1.0, g1 / np.maximum(n1, 1e-12), g1)
        g2 = np.where(n2 > 1.0, g2 / np.maximum(n2, 1e-12), g2)
        grad_f1 = g1.sum(axis=0)
        grad_f2 = g2.sum(axis=0)
        sep = F1 - F2
        grad_f1 += 2.0 * lambda_sep * sep
        grad_f2 -= 2.0 * lambda_sep * sep
        nf1 = np.linalg.norm(grad_f1)
        nf2 = np.linalg.norm(grad_f2)
        if nf1 > 5.0:
            grad_f1 *= 5.0 / nf1
        if nf2 > 5.0:
            grad_f2 *= 5.0 / nf2
        F1 -= lr * grad_f1
        F2 -= lr * grad_f2
        if not (np.all(np.isfinite(F1)) and np.all(np.isfinite(F2))):
            break
    return F1, F2


def refine_moduli_by_axis_fast(pts, r30, r45, a, t10, t20, pca_axis, max_iter=50,
                               angle_tol_deg=0.5, fd_h=1e-3,
                               w_dir=20.0, w_self=1.0, w_theory=1e-3):
    n = len(pts)
    axis = np.asarray(pca_axis, float)
    axis = axis / np.linalg.norm(axis)
    x = np.array([t10.real, t10.imag, t20.real, t20.imag], float)
    lam = 1e-3
    F1 = F2 = np.zeros(3)

    def residuals(xx):
        t1 = complex(xx[0], xx[1])
        t2 = complex(xx[2], xx[3])
        pt, p1, p2, sig = dd.fast_probs(r30, r45, t1, t2, a)
        F1c, F2c = m.extract_foci(pts, p1, sig[1], p2, sig[2])
        r = np.zeros(n + 9)
        inv = 1.0 / math.sqrt(n)
        P = np.asarray(pts, float)
        d1 = np.linalg.norm(P - F1c, axis=1)
        d2 = np.linalg.norm(P - F2c, axis=1)
        delta = d1 + d2 - 2.0 * a
        prob_foci = np.exp(-np.abs(delta) / (2.0 * a))
        r[:n] = w_self * (pt - prob_foci) * inv
        dv = F1c - F2c
        if np.linalg.norm(dv) < 1e-12:
            dv = axis
        dv = dv / np.linalg.norm(dv)
        if np.dot(dv, axis) < 0:
            dv = -dv
        e = dv - axis
        r[n] = w_dir * e[0]
        r[n + 1] = w_dir * e[1]
        r[n + 2] = w_dir * e[2]
        r[n + 3] = w_theory * (xx[0] - t10.real)
        r[n + 4] = w_theory * (xx[1] - t10.imag)
        r[n + 5] = w_theory * (xx[2] - t20.real)
        r[n + 6] = w_theory * (xx[3] - t20.imag)
        im1 = xx[1]
        dev1 = 0
        for i in range(100):
            if im1 - 2 ** dev1 < 0:
                break
            dev1 += 1
        dy = 1.0
        for i in range(int(dev1)):
            dy += 1.0 / (2 ** i)
        r[n + 7] = dy * (xx[1] - 1)
        r[n + 8] = dy * (xx[3] - 1)
        return r, F1c, F2c

    for it in range(max_iter):
        r, F1, F2 = residuals(x)
        cost = float(np.sum(r ** 2))
        dv = F1 - F2
        ang = 180.0
        if np.linalg.norm(dv) > 1e-12:
            dv = dv / np.linalg.norm(dv)
            if np.dot(dv, axis) < 0:
                dv = -dv
            ang = angle(dv, axis)
        if ang < angle_tol_deg:
            break
        J = np.zeros((len(r), 4))
        for k in range(4):
            xp = x.copy()
            xp[k] += fd_h
            rp2, _, _ = residuals(xp)
            J[:, k] = (rp2 - r) / fd_h
        A = J.T @ J
        g = J.T @ r
        Aaug = A.copy()
        for c in range(4):
            Aaug[c, c] += lam * (A[c, c] + 1e-12)
        delta = np.linalg.solve(Aaug, g)
        xtry = x - delta
        r2, _, _ = residuals(xtry)
        cost2 = float(np.sum(r2 ** 2))
        if cost2 < cost:
            x = xtry
            lam = max(lam / 3.0, 1e-8)
        else:
            lam = min(lam * 3.0, 1e4)
    r, F1, F2 = residuals(x)
    return complex(x[0], x[1]), complex(x[2], x[3]), F1, F2


def nm_on_delta(t1_init, t2_init, pts, r30, r45, a, pcav, f1f2v, max_evals=600, stop_f=10.0):
    lb = np.array([-0.5, 0.5, -0.5, 0.5])
    ub = np.array([0.5, 1.5, 0.5, 1.5])
    evals = [0]
    last = {}

    def objective(xx):
        evals[0] += 1
        t1 = complex(xx[0], xx[1])
        t2 = complex(xx[2], xx[3])
        pt, p1, p2, sig = dd.fast_probs(r30, r45, t1, t2, a)
        F1, F2 = m.extract_foci(pts, p1, sig[1], p2, sig[2])
        f1, f2 = fit_foci_fast(pts, pt, F1, F2, a)
        d = f1 - f2
        last["F1z"], last["F2z"] = f1, f2
        if np.dot(d, d) < 1e-12:
            return 0.0
        d = d / np.linalg.norm(d)
        ad = angle(d, pcav)
        ad1 = angle(d, f1f2v)
        last["ad"], last["ad1"] = ad, ad1
        return abs(ad - ad1)

    x0 = np.clip([t1_init.real, t1_init.imag, t2_init.real, t2_init.imag], lb, ub)

    def cb(xk):
        return objective(xk) <= stop_f

    res = minimize(objective, x0, method="Nelder-Mead", bounds=list(zip(lb, ub)),
                   callback=cb, options={"maxiter": max_evals, "maxfev": max_evals * 2,
                                         "xatol": 1e-6, "fatol": 1e-6})
    best = np.clip(res.x, lb, ub)
    objective(best)
    return (complex(best[0], best[1]), complex(best[2], best[3]),
            last.get("ad", 180.0), last.get("ad1", 180.0), evals[0],
            last.get("F1z"), last.get("F2z"))


def run(name, pts, T_field=0.59, max_outer=8):
    P = np.array(pts, float)
    rp = n2.compute_rp([np.array(p, float) for p in pts])
    d2 = math.asin(math.cos(math.radians(30)) / math.pi)
    a = rp * (1 + math.sin(d2))
    e3 = math.cos(d2) * 2 / (1 + math.sin(d2))
    c = rp * math.cos(d2) * e3
    _, r45, r30, _ = n2.yzqx([np.array(p, float) for p in pts], rp)
    t1t, t2t = n2.compute_taus()
    t_ = complex(0, 1)
    pt, p1, p2, sig = dd.fast_probs(r30, r45, t1t, t2t, a)
    F1, F2 = m.extract_foci(pts, p1, sig[1], p2, sig[2])
    f1, f2 = fit_foci_fast(pts, pt, F1, F2, a)
    pt01, p101, p201, sig01 = dd.fast_probs(r30, r45, t_, t_, a)
    F01, F02 = m.extract_foci(pts, p101, sig01[1], p201, sig01[2])
    v = m.pca(pts)[1]
    v = v / np.linalg.norm(v)

    def norm(vv):
        return np.asarray(vv, float) / np.linalg.norm(vv)

    v1 = norm(F1 - F2)
    v3 = norm(F01 - F02)
    s1 = batch_probability(P, v1 * c, -v1 * c, a)
    s3 = batch_probability(P, v3 * c, -v3 * c, a)

    vj = v + norm(F1 - F2)
    refine_moduli_by_axis_fast(pts, r30, r45, a, t_, t_, vj, 50)
    t1, t2 = t_, t_
    f1z, f2z = f1, f2
    print(f"  [{name}] ∠(v,v1)={angle(v, v1):.1f}°")
    for i in range(max_outer):
        d = norm(f1z - f2z)
        ad = angle(d, v)
        ad1 = angle(d, v1)
        delta = abs(ad - ad1)
        cond_ang = (ad + ad1) * 0.5 - min(ad, ad1)
        s5 = batch_probability(P, d * c, -d * c, a)
        dc = abs(corr(s5, s1) - corr(s5, s3))
        a_ok = cond_ang <= 8
        p_ok = dc < T_field
        print(f"    iter{i}: ∠(d,v)={ad:6.1f} ∠(d,v1)={ad1:6.1f} |Δ|={delta:5.1f} "
              f"ang{'✅' if a_ok else ''} | dc={dc:.3f} field{'✅' if p_ok else ''}",
              flush=True)
        if a_ok:
            break
        if i >= max_outer - 1:
            break
        t1n, t2n, F1n, F2n = refine_moduli_by_axis_fast(
            pts, r30, r45, a, t_, t_, norm(f1z - f2z), 50)
        axis = np.cross(norm(F1n - F2n), norm(F1 - F2))
        axis1 = np.cross(norm(F1n - F2n), norm(F01 - F02))
        newv = norm(F01 - F02) if np.linalg.norm(axis) < 1e-9 else None
        # 简化: 用旋转 (与用户一致)
        def rotate(vv, ax, deg):
            if np.linalg.norm(ax) < 1e-12:
                return vv
            ax = ax / np.linalg.norm(ax)
            rg = math.radians(deg)
            return vv * math.cos(rg) + np.cross(ax, vv) * math.sin(rg) \
                + ax * np.dot(ax, vv) * (1 - math.cos(rg))
        newv = rotate(norm(F01 - F02), axis, -ad)
        newF = rotate(norm(F1 - F2), axis1, -ad)
        t1, t2, ad_, ad1_, ev, F1z, F2z = nm_on_delta(
            t1n, t2n, pts, r30, r45, a, newv, newF)
        f1z, f2z = F1z, F2z
    print()


def main():
    datasets = {}
    with open(PYJSON, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    datasets["pyjson"] = [np.array([p["x"], p["y"], p["z"]], float) for p in raw]
    datasets["ellip0"] = dd.make_ellip(0)
    datasets["ball0"] = dd.make_ball(0)
    print("=" * 90)
    print("真实迭代: 角度条件 vs 场条件 |corr(s5,s1)-corr(s5,s3)|<0.59")
    print("=" * 90)
    for name, pts in datasets.items():
        try:
            run(name, pts)
        except Exception as e:
            print(f"  {name}: 失败 ({e})")
    print("DONE")


if __name__ == "__main__":
    main()
