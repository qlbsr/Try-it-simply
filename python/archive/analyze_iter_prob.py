# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
在真实迭代轨迹内找"概率条件 → 角度终止条件"的替代关系 (完全脱离 PCA):
  复现 n2sjy2.cs 每轮: 记录 angleDeg, angleDeg1, |Δ|, 以及 s1..s4 (不依赖 PCA):
    s1 = P((F1-F2)方向)    理论 taus 焦点对
    s2 = P((f1-f2)方向)    拟合焦点对
    s3 = P((F01-F02)方向)  (0,1) 焦点对
    s4 = P((f01-f02)方向)  (0,1) 拟合焦点对
    s5 = P((f1z-f2z)方向)  当前迭代焦点对
  统计: |Δ| 与 |s_i - s_j|×100 的关系 → 总结条件式
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


def main():
    with open(PYJSON, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    pts = [np.array([p["x"], p["y"], p["z"]], float) for p in raw]
    P = np.array(pts, float)
    rp = n2.compute_rp(pts)
    d2 = math.asin(math.cos(math.radians(30)) / math.pi)
    a = rp * (1 + math.sin(d2))
    e3 = math.cos(d2) * 2 / (1 + math.sin(d2))
    h = rp * math.cos(d2)
    c = h * e3
    _, r45, r30, _ = n2.yzqx(pts, rp)
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
    v2 = norm(f1 - f2)
    v3 = norm(F01 - F02)

    # 迭代 (简化: 只跑前几轮, 每轮记录概率与角度)
    t1, t2 = t_, t_
    f1z, f2z = f1, f2
    print(f"{'iter':>4s} {'∠(d,v)':>7s} {'∠(d,v1)':>7s} {'|Δ|':>6s} "
          f"{'s1':>7s} {'s2':>7s} {'s3':>7s} {'s5':>7s} "
          f"{'|s1-s5|×100':>11s} {'|s2-s5|×100':>11s} {'|s3-s5|×100':>11s} "
          f"{'|s1-s3|×100':>11s}")
    rows = []
    for i in range(8):
        d = norm(f1z - f2z)
        ad = angle(d, v)
        ad1 = angle(d, v1)
        delta = abs(ad - ad1)
        s1 = batch_probability(P, v1 * c, -v1 * c, a)[0]
        s2 = batch_probability(P, v2 * c, -v2 * c, a)[0]
        s3 = batch_probability(P, v3 * c, -v3 * c, a)[0]
        s5 = batch_probability(P, d * c, -d * c, a)[0]
        rows.append((delta, abs(s1 - s5) * 100, abs(s2 - s5) * 100,
                     abs(s3 - s5) * 100, abs(s1 - s3) * 100))
        print(f"{i:4d} {ad:7.1f} {ad1:7.1f} {delta:6.1f} "
              f"{s1*100:7.2f} {s2*100:7.2f} {s3*100:7.2f} {s5*100:7.2f} "
              f"{abs(s1-s5)*100:11.2f} {abs(s2-s5)*100:11.2f} {abs(s3-s5)*100:11.2f} "
              f"{abs(s1-s3)*100:11.2f}", flush=True)
        if delta <= 16:
            print(f"== 角度条件达成 (|Δ|={delta:.1f} ≤ 16), 迭代 {i} ==")
            break
        # 下一步: NM (同用户流程)
        t1n, t2n, F1n, F2n = t1, t2, F1, F2
        axis = np.cross(norm(F1n - F2n), norm(F1 - F2))
        axis1 = np.cross(norm(F1n - F2n), norm(F01 - F02))
        newv = rotate_vec(norm(F01 - F02), axis, -ad)
        newF = rotate_vec(norm(F1 - F2), axis1, -ad)
        t1, t2, ad_, ad1_, ev, F1z, F2z = nm_on_delta(t1n, t2n, pts, r30, r45, a, newv, newF)
        f1z, f2z = F1z, F2z
    R = np.array(rows)
    print()
    print("=" * 70)
    print("相关性: |Δ| vs 各概率差×100")
    print("=" * 70)
    for k, lab in enumerate(["|s1-s5|", "|s2-s5|", "|s3-s5|", "|s1-s3|"]):
        corr = np.corrcoef(R[:, 0], R[:, k + 1])[0, 1] if R[:, k + 1].std() > 0 else float('nan')
        print(f"  corr(|Δ|, {lab}×100) = {corr:+.3f}")
    print("DONE")


def rotate_vec(v, axis, deg):
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return v.copy()
    axis = axis / n
    ang = math.radians(deg)
    return (v * math.cos(ang) + np.cross(axis, v) * math.sin(ang)
            + axis * np.dot(axis, v) * (1 - math.cos(ang)))


if __name__ == "__main__":
    main()
