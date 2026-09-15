# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
改善 LM 活性: axis=当前d(自指) 一步对齐→活性死→无法推进
  方案: axis 逐步逼近外部参考 (slerp), 保持每轮 angleDeg 活跃 (非0)
  策略:
    A: axis = d              (现状, 自指 → 一步到位, 活性死)
    B: axis = slerp(d, ref, w)   (朝参考 ref 部分推进, angleDeg 每轮活跃)
  测试: 外环仍用旋转+NM; 看 a9=∠(d,v) 是否单调推进
  参考 ref: 用固定格点方向族 (无PCA优先), 也测 v (对照)
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


def slerp(u, w, t):
    u = u / np.linalg.norm(u)
    w = w / np.linalg.norm(w)
    dot = np.clip(np.dot(u, w), -1, 1)
    theta = math.acos(dot)
    if theta < 1e-9:
        return u.copy()
    su = math.sin((1 - t) * theta) / math.sin(theta)
    sw = math.sin(t * theta) / math.sin(theta)
    v = su * u + sw * w
    return v / np.linalg.norm(v)


def refine_lm_axis(pts, r30, r45, a, t10, t20, axis_v, n_iter=50):
    """RefineModuliByAxis: LM 对齐到给定 axis (axis 由外部决定, 非自指)"""
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


def rotate(vv, a_, deg):
    if np.linalg.norm(a_) < 1e-12:
        return vv
    a_ = a_ / np.linalg.norm(a_)
    rg = math.radians(deg)
    return vv * math.cos(rg) + np.cross(a_, vv) * math.sin(rg) \
        + a_ * np.dot(a_, vv) * (1 - math.cos(rg))


def run_strategy(name, pts, mode="A", omega_lm=0.5, ref_kind="v2", n_rounds=40):
    """
    mode A: axis=d (自指, 现状)
    mode B: axis=slerp(d, ref, omega_lm)  每次 LM 朝 ref 推进 omega_lm 比例
    ref_kind: v2 (理论拟合, 无PCA) / v4 ((0,1)拟合) / mid (v2,v4中点) / v (PCA对照)
    """
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
    v = m.pca(pts)[1]
    v = v / np.linalg.norm(v)

    def norm(vv):
        return np.asarray(vv, float) / np.linalg.norm(vv)

    v2 = norm(f1 - f2)
    v4 = norm(f01 - f02)
    if ref_kind == "v2":
        ref = v2
    elif ref_kind == "v4":
        ref = v4
    elif ref_kind == "mid":
        ref = norm(v2 + v4)
    else:
        ref = v

    t1, t2 = t_, t_
    f1z, f2z = f1, f2
    a9_min = 999.0
    traj = []
    for i in range(n_rounds):
        d = norm(f1z - f2z)
        # LM 内层: axis 策略
        if mode == "A":
            axis_v = d                     # 自指
        else:
            axis_v = slerp(d, ref, omega_lm)   # 朝 ref 推进 omega_lm
        t1n, t2n, F1n, F2n = refine_lm_axis(pts, r30, r45, a, t_, t_, axis_v)
        ptn, p1n, p2n, sgn = dd.fast_probs(r30, r45, t1n, t2n, a)
        f1n, f2n = fit_fast(P, ptn, F1n, F2n, a)
        v6 = norm(f1n - f2n)
        a9 = angle(v6, v)
        a9_min = min(a9_min, a9)
        traj.append(a9)
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
    return a9_min, traj


def main():
    print("=" * 100)
    print("LM 活性改善测试: 外环 a9=∠(d,v) 历史最小 (40轮)")
    print("=" * 100)
    for name, pts in [("ball2", dd.make_ball(2)), ("ellip0", dd.make_ellip(0)),
                      ("ball3", dd.make_ball(3))]:
        print(f"  [{name}]")
        for mode, desc in [("A", "axis=d 自指(现状)"), ("B", "axis=slerp(d,ref,0.3)"),
                           ("B2", "axis=slerp(d,ref,0.6)")]:
            for rk in (["v2", "v"] if mode == "A" else ["v2", "v4", "mid", "v"]):
                try:
                    w = 0.3 if mode == "B" else 0.6
                    am, tr = run_strategy(name, pts, mode="A" if mode == "A" else "B",
                                          omega_lm=w, ref_kind=rk)
                    tag = desc if rk in ("v2",) and mode != "A" else f"{desc} ref={rk}"
                    print(f"    {tag:42s}: a9_min={am:5.1f}°  末={tr[-1]:5.1f}°", flush=True)
                except Exception as e:
                    print(f"    {rk}: 失败 {e}", flush=True)
        print(flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
