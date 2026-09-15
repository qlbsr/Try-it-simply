# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
核心问题: 能否完全脱离 PCA?
  方案: 全程不调用 pca()/协方差 — 迭代参考全用焦点方向 (v1/v3/v2/v4, 格点提取)
        预精化 vj 改用纯焦点方向 (不用 v)
  验证: ① 无 PCA 判据 #2 是否仍达成 (收敛性不受影响)
        ② 结束时当前方向 v6 与"真协方差主轴"的对齐角 (仅事后测量)
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


def nm_refine(pts, r30, r45, a, t1, t2, ref1, ref2, max_evals=600):
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
        return obj(xk) <= 10.

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


def run_pca_free(name, pts, max_outer=20, use_vj_focus=True):
    """完全无 PCA 流程: 不调用 pca(), 预精化 vj 用 Fd (或纯 (0,1)方向)"""
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
    # ★ 不调用 pca()! 仅事后用真协方差主轴测量 (不算迭代输入)
    C = P.T @ P
    evals, evecs = np.linalg.eigh(C)
    v_true = evecs[:, np.argsort(evals)[::-1][0]]   # 真主轴 (仅测量)

    def norm(vv):
        return np.asarray(vv, float) / np.linalg.norm(vv)

    v2 = norm(f1 - f2)          # 理论拟合焦点方向
    v4 = norm(f01 - f02)        # (0,1) 拟合焦点方向
    Fd = norm(F1 - F2)
    # 预精化: vj 用纯焦点组合 (替代原 v+Fd)
    if use_vj_focus:
        vj = Fd + norm(F01 - F02)     # 两个格点提取方向之和 (无 PCA)
    else:
        vj = norm(f1 - f2) + norm(f01 - f02)
    t1, t2 = t_, t_
    f1z, f2z = f1, f2
    best_align = 999.0
    best_d = None
    ok = False
    for i in range(max_outer):
        t1n, t2n, F1n, F2n = refine_lm(pts, r30, r45, a, t_, t_, norm(f1z - f2z))
        ptn, p1n, p2n, sgn = dd.fast_probs(r30, r45, t1n, t2n, a)
        f1n, f2n = fit_fast(P, ptn, F1n, F2n, a)
        v6 = norm(f1n - f2n)
        ang5 = angle(v6, v2)      # ∠(v6, v2)
        ang6 = angle(v6, v4)      # ∠(v6, v4)
        cond = abs((ang5 + ang6) * 0.5 - min(ang5, ang6))
        align = angle(v6, v_true)  # 仅测量
        if align < best_align:
            best_align = align
            best_d = v6
        if i % 4 == 0 or cond <= 8:
            print(f"    iter{i:2d}: cond#2={cond:5.1f} {'✅' if cond<=8 else ''} "
                  f"| v6 vs 真主轴={align:5.1f}° (best {best_align:5.1f}°)", flush=True)
        if cond <= 8:
            ok = True
            break
        if i >= max_outer - 1:
            break
        # 旋转参考 (无 PCA)
        axis = np.cross(v6, v2)
        axis1 = np.cross(v6, v4)
        newf = rotate(v2, axis, -ang5)
        newv = rotate(v4, axis1, -ang6)
        t1, t2 = nm_refine(pts, r30, r45, a, t1n, t2n, newf, newv)
        ptn, p1n, p2n, sgn = dd.fast_probs(r30, r45, t1, t2, a)
        F1n, F2n = m.extract_foci(pts, p1n, sgn[1], p2n, sgn[2])
        f1z, f2z = fit_fast(P, ptn, F1n, F2n, a)
    print(f"    ==> {'达成' if ok else '未达成'} | best v6 vs 真主轴 = {best_align:.1f}° "
          f"{'✅≤10°' if best_align<=10 else '✗>10°'}")
    return ok, best_align


def main():
    import json
    datasets = {}
    with open(r"C:\Users\23128\My project (2)\Assets\Resources\pyjson.json", encoding="utf-8") as f:
        raw = json.load(f)
    datasets["pyjson"] = [np.array([p["x"], p["y"], p["z"]], float) for p in raw]
    for seed in range(4):
        datasets[f"ball{seed}"] = dd.make_ball(seed)
    for seed in range(3):
        datasets[f"ellip{seed}"] = dd.make_ellip(seed)
    for seed in range(2):
        datasets[f"cube{seed}"] = dd.make_cube(10 + seed)
    print("=" * 96)
    print("完全无 PCA 流程 (不调用 pca()): 判据#2达成? v6对齐真主轴?")
    print("=" * 96)
    ok_n = 0
    align_n = 0
    for name, pts in datasets.items():
        try:
            ok, align = run_pca_free(name, pts)
            if ok:
                ok_n += 1
            if align <= 10:
                align_n += 1
        except Exception as e:
            print(f"  {name}: 失败 ({e})", flush=True)
        print(flush=True)
    print(f"达成率 {ok_n}/10 | v6对齐真主轴≤10° 数据集数 {align_n}/10")
    print("DONE")


if __name__ == "__main__":
    main()
