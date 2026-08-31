# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
测试用户改进后的 n2sjy2.cs 逻辑 (忠实移植, Python 版):
  - 数据: 真实 pyjson.json (200 点)
  - vj = v̂ + d̂  (两向量分别归一化后相加, 不归一化)
  - 主循环注释掉 FitFociByProbability (成本 -100×)
  - 角度用 (f1-f2) 计算; 停止条件 (a+b)/2 - min(a,b) <= 5
  - RefineTausWithNM: Nelder-Mead 最小化 |Δ|, bounds [±0.5,0.5]×[0.5,1.5], stopF=10
计时: 目标验证用户"几乎 20s 内达到条件"
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
    """Vector3.Angle: 两向量夹角 0~180°"""
    un = np.linalg.norm(u)
    vn = np.linalg.norm(v)
    if un < 1e-12 or vn < 1e-12:
        return 0.0
    return math.degrees(math.acos(np.clip(np.dot(u, v) / (un * vn), -1, 1)))


def rotate(v, axis, angle_deg):
    """Quaternion.AngleAxis(angle, axis) * v (Rodrigues)"""
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return v.copy()
    axis = axis / n
    ang = math.radians(angle_deg)
    return (v * math.cos(ang)
            + np.cross(axis, v) * math.sin(ang)
            + axis * np.dot(axis, v) * (1 - math.cos(ang)))


def fit_foci_fast(points, true_prob, init_f1, init_f2, a,
                  max_iter=300, lr=0.0001, lambda_sep=1.0):
    """FitFociByProbability 的 numpy 向量化版 (算法一致)"""
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
        sign = np.where(delta >= 0, 1.0, -1.0)
        coef = diff * fit_prob * sign / (2.0 * a)
        g1 = coef[:, None] * (P - F1) / sd1[:, None]
        g2 = coef[:, None] * (P - F2) / sd2[:, None]
        # 单点梯度裁剪
        n1 = np.linalg.norm(g1, axis=1, keepdims=True)
        n2 = np.linalg.norm(g2, axis=1, keepdims=True)
        g1 = np.where(n1 > 1.0, g1 / np.maximum(n1, 1e-12), g1)
        g2 = np.where(n2 > 1.0, g2 / np.maximum(n2, 1e-12), g2)
        grad_f1 = g1.sum(axis=0)
        grad_f2 = g2.sum(axis=0)
        sep = F1 - F2
        grad_f1 += 2.0 * lambda_sep * sep
        grad_f2 -= 2.0 * lambda_sep * sep
        # 整体裁剪
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
    """C# RefineModuliByAxis 忠实移植 (用 fast_probs 替代纯 Python 概率)"""
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
        # C#: dy 软约束拉向 (0,1)
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
        # 数值雅可比
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


def refine_taus_with_nm(t1_init, t2_init, pts, r30, r45, a, pcav, f1f2v,
                        max_evals=600, stop_f=10.0):
    """C# RefineTausWithNM 忠实移植: NM 最小化 |Δ|, bounds [±0.5,0.5]×[0.5,1.5]"""
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
    final_f = objective(best)
    return (complex(best[0], best[1]), complex(best[2], best[3]),
            final_f, last.get("ad", 180.0), last.get("ad1", 180.0),
            evals[0], last.get("F1z"), last.get("F2z"))


def main():
    with open(PYJSON, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    points = [np.array([p["x"], p["y"], p["z"]], float) for p in raw]
    print(f"数据: {len(points)} 点 (pyjson.json)")

    t_start = time.perf_counter()
    t1, t2 = n2.compute_taus()
    rp = n2.compute_rp(points)
    d2 = math.asin(math.cos(math.radians(30)) / math.pi)
    a = rp * (1 + math.sin(d2))
    _, r45, r30, _ = n2.yzqx(points, rp)

    v = m.pca(points)[1]            # C# pca() 返回 v3 (重建向量)
    v = v / np.linalg.norm(v)

    # 理论 taus 下的初始焦点
    pt, p1, p2, sig = dd.fast_probs(r30, r45, t1, t2, a)
    F1, F2 = m.extract_foci(points, p1, sig[1], p2, sig[2])
    f1, f2 = fit_foci_fast(points, pt, F1, F2, a)

    # t_ = (0,1) 下的参考焦点
    t_ = complex(0, 1)
    pt01, p101, p201, sig01 = dd.fast_probs(r30, r45, t_, t_, a)
    F01, F02 = m.extract_foci(points, p101, sig01[1], p201, sig01[2])
    f01, f02 = fit_foci_fast(points, pt01, F01, F02, a)

    # 用户改动: vj = v̂ + d̂ (分别归一化, 不归一化)
    vj = v / np.linalg.norm(v) + (F1 - F2) / np.linalg.norm(F1 - F2)
    (t1_01, t2_01, F1_n, F2_n) = refine_moduli_by_axis_fast(points, r30, r45, a, t_, t_, vj, 50)
    angle_degv = angle(vj, v / np.linalg.norm(v))
    print(f"vj 夹角(与v): {angle_degv:.2f}° | 预精化 t1={t1_01:.4f} t2={t2_01:.4f}")

    t_loop = time.perf_counter()
    for i in range(300):
        pt_new, _, _, _ = dd.fast_probs(r30, r45, t1, t2, a)
        # 用户改动: 方向目标 = (f1-f2).normalized (非 vj)
        t1_new, t2_new, F1_new, F2_new = refine_moduli_by_axis_fast(
            points, r30, r45, a, t_, t_, (f1 - f2) / np.linalg.norm(f1 - f2), 50)
        # 用户改动: 注释掉 fit_foci_by_probability (不再每轮重拟合)
        angle_deg = angle(f1 - f2, v)
        angle_deg1 = angle(f1 - f2, F1 - F2)
        cond = (angle_deg + angle_deg1) * 0.5 - min(angle_deg, angle_deg1)
        print(f"iter {i:3d}: ∠(f1-f2,v)={angle_deg:7.2f}° ∠(f1-f2,F1-F2)={angle_deg1:7.2f}° "
              f"∠v={angle_degv:6.2f}° cond={cond:6.2f}  [{(time.perf_counter()-t_loop):5.1f}s]")
        if cond <= 5:
            print(f"== 达成条件 (|Δ|/2={cond:.2f} ≤ 5), 迭代 {i} 次 ==")
            break
        axis = np.cross((F1_new - F2_new) / np.linalg.norm(F1_new - F2_new),
                        (F1 - F2) / np.linalg.norm(F1 - F2))
        axis1 = np.cross((F1_new - F2_new) / np.linalg.norm(F1_new - F2_new),
                         (F01 - F02) / np.linalg.norm(F01 - F02))
        newv = rotate((F01 - F02) / np.linalg.norm(F01 - F02), axis, -angle_deg)
        newF = rotate((F1 - F2) / np.linalg.norm(F1 - F2), axis1, -angle_deg)
        (t1_, t2_, delta_deg, ad_, ad1_, evals, F1z, F2z) = refine_taus_with_nm(
            t1_new, t2_new, points, r30, r45, a, newv, newF)
        t1, t2 = t1_, t2_
        f1, f2 = F1z, F2z
        print(f"        NM: |Δ|={delta_deg:.2f}° (∠={ad_:.1f}°,∠1={ad1_:.1f}°, {evals}次求值) "
              f"t1={t1:.4f} t2={t2:.4f}")

    total = time.perf_counter() - t_start
    print(f"\nTOTAL {total:.1f}s (循环 {time.perf_counter()-t_loop:.1f}s)  DONE")


if __name__ == "__main__":
    main()
