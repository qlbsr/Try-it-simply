# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
GoldenSpiralDirection 集成测试 (用户方案):
  当 angleDegnew < 5 (方向几乎对齐参考, 活性不足) →
  用黄金角螺旋从当前夹角逐档增长到 60°, 均匀遍历候选方向
  每个候选方向作为 LM 的 axis, 试探哪个能激活收敛 (a9 下降 / 越过平台)
复现 C# GoldenSpiralDirection.GenerateSpiralDirections
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


def golden_spiral_dirs(current_dir, r, target_angle_deg=60.0, angle_step_deg=3.0):
    """C# GoldenSpiralDirection 复现"""
    current_dir = np.asarray(current_dir, float)
    current_dir = current_dir / np.linalg.norm(current_dir)
    r = np.asarray(r, float)
    r = r / np.linalg.norm(r)
    current_angle = angle(current_dir, r)
    if target_angle_deg <= current_angle:
        return []
    steps = int(math.ceil((target_angle_deg - current_angle) / angle_step_deg))
    u = np.cross(r, current_dir)
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(r, np.array([1., 0, 0]))
    u = u / np.linalg.norm(u)
    w = np.cross(r, u)
    w = w / np.linalg.norm(w)
    phi0 = math.atan2(np.dot(current_dir, w), np.dot(current_dir, u))
    golden = math.radians(137.507764)
    dirs = []
    for i in range(steps):
        theta = math.radians(min(current_angle + (i + 1) * angle_step_deg, target_angle_deg))
        phi = phi0 + i * golden
        d = math.cos(theta) * r + math.sin(theta) * (math.cos(phi) * u + math.sin(phi) * w)
        dirs.append(d / np.linalg.norm(d))
    return dirs


def fit_fast(P, prob, F1, F2, a, iters=100, lr=0.0001):
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


def nm_refine(pts, r30, r45, a, t1, t2, ref1, ref2, max_evals=50):
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


def rotate(vv, ax, deg):
    ax = np.asarray(ax, float)
    if np.linalg.norm(ax) < 1e-12:
        return vv.copy()
    ax = ax / np.linalg.norm(ax)
    rg = math.radians(deg)
    v = vv * math.cos(rg) + np.cross(ax, vv) * math.sin(rg) \
        + ax * np.dot(ax, vv) * (1 - math.cos(rg))
    return v / np.linalg.norm(v)


def run_pyjson(mode, max_outer=60, step=3.0):
    """mode: 'spiral' 螺旋遍历 | 'random60' 随机60 | 'none' 纯自然"""
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
    v3 = m.pca(pts)[1]
    v3 = v3 / np.linalg.norm(v3)
    C = P.T @ P
    evals, evecs = np.linalg.eigh(C)
    v_true = evecs[:, np.argsort(evals)[::-1][0]]

    def norm(vv):
        return np.asarray(vv, float) / np.linalg.norm(vv)

    v2 = norm(f1 - f2)
    v4 = norm(f01 - f02)
    f01fit = norm(f01 - f02)
    rng = np.random.default_rng(1)
    t1, t2 = t_, t_
    f1z, f2z = f1, f2
    best_a9v, best_a9t = 999.0, 999.0
    spiral_budget = 6  # 每次停滞最多试几个螺旋方向
    for i in range(max_outer):
        d_cur = norm(f1z - f2z)
        angNew = angle(f01fit, d_cur)
        axi = d_cur
        if mode == "spiral" and angNew < 5:
            # 螺旋遍历: 从当前角逐步到60°, 试前几个方向, 选使 LM 后 a9 最小的
            dirs = golden_spiral_dirs(d_cur, f01fit, 60.0, step)[:spiral_budget]
            best_ax, best_score = None, 1e18
            for cand in dirs:
                # 试探: 用候选方向做 LM, 看结果方向与 v3 夹角(越小越推进)
                tn1, tn2, Fn1, Fn2 = refine_lm(pts, r30, r45, a, t_, t_, cand)
                ptn, pn1, pn2, sgn = dd.fast_probs(r30, r45, tn1, tn2, a)
                fn1, fn2 = fit_fast(P, ptn, Fn1, Fn2, a)
                vv6 = norm(fn1 - fn2)
                sc = angle(vv6, v3)
                if sc < best_score:
                    best_score = sc
                    best_ax = cand
            if best_ax is not None:
                axi = best_ax
        elif mode == "random60" and angNew < 5:
            dPerp = d_cur - np.dot(d_cur, f01fit) * f01fit
            if np.linalg.norm(dPerp) > 1e-6:
                dPerp = dPerp / np.linalg.norm(dPerp)
                onCone = 0.5 * f01fit + 0.866 * dPerp
                az = rng.uniform(0, 2 * math.pi)
                axi = rotate(onCone, f01fit, math.degrees(az))
        t1n, t2n, F1n, F2n = refine_lm(pts, r30, r45, a, t_, t_, axi)
        ptn, p1n, p2n, sgn = dd.fast_probs(r30, r45, t1n, t2n, a)
        f1n, f2n = fit_fast(P, ptn, F1n, F2n, a)
        v6 = norm(f1n - f2n)
        a9v = angle(v6, v3)
        a9t = angle(v6, v_true)
        best_a9v = min(best_a9v, a9v)
        best_a9t = min(best_a9t, a9t)
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
    return best_a9v, best_a9t


def main():
    print("=" * 90)
    print("pyjson: 黄金螺旋遍历 vs 随机60 vs 纯自然 (60外层轮)")
    print("=" * 90)
    for mode, desc in [("none", "纯自然"), ("random60", "随机60°扰动"),
                       ("spiral", "黄金螺旋遍历(步长3°)")]:
        try:
            av, at = run_pyjson(mode)
            print(f"  {desc:18s}: a9(v3)={av:6.1f}°  a9(真)={at:6.1f}°", flush=True)
        except Exception as e:
            print(f"  {desc}: 失败 {e}", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
