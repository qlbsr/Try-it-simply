# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
最终普遍性验证: 忠实移植当前 n2sjy2.cs (无终止, 80轮, 每轮 a9=∠(v6,v))
  统计: 纯靠迭代次数, a9 历史最小能达到多少? (不做终止, 看次数是否足够)
"""
import math
import numpy as np

import data_driven_axis as dd
import n2sjy2 as n2
import nsjy_algorithms as m


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


def run_n2sjy2(name, pts, n_rounds=80):
    """忠实移植当前 n2sjy2.cs Start(): 无终止条件, n_rounds 轮, 每轮记录 a9"""
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

    t1, t2 = t_, t_
    f1z, f2z = f1, f2
    a9_min = 999.0
    a9_first = None
    a9_last = None
    for i in range(n_rounds):
        t1n, t2n, F1n, F2n = refine_lm(pts, r30, r45, a, t_, t_, norm(f1z - f2z))
        ptn, p1n, p2n, sgn = dd.fast_probs(r30, r45, t1n, t2n, a)
        f1n, f2n = fit_fast(P, ptn, F1n, F2n, a)
        v6 = norm(f1n - f2n)
        a9 = angle(v6, v)
        if a9_first is None:
            a9_first = a9
        a9_last = a9
        a9_min = min(a9_min, a9)
        ang5 = angle(v6, v2)
        ang6 = angle(v6, v4)
        # 旋转 (用户代码)
        axis = np.cross(v6, v2)
        axis1 = np.cross(v6, v4)
        newf = rotate(v2, axis, -ang5)
        newv = rotate(v4, axis1, -ang6)
        t1, t2 = nm_refine(pts, r30, r45, a, t1n, t2n, newf, newv)
        ptn, p1n, p2n, sgn = dd.fast_probs(r30, r45, t1, t2, a)
        F1n, F2n = m.extract_foci(pts, p1n, sgn[1], p2n, sgn[2])
        f1z, f2z = fit_fast(P, ptn, F1n, F2n, a)
    return a9_first, a9_last, a9_min


def main():
    import json
    datasets = {}
    with open(r"C:\Users\23128\My project (2)\Assets\Resources\pyjson.json", encoding="utf-8") as f:
        raw = json.load(f)
    datasets["pyjson"] = [np.array([p["x"], p["y"], p["z"]], float) for p in raw]
    for seed in range(4):
        datasets[f"ball{seed}"] = dd.make_ball(seed)
    for seed in range(4):
        datasets[f"ellip{seed}"] = dd.make_ellip(seed)
    for seed in range(3):
        datasets[f"cube{seed}"] = dd.make_cube(10 + seed)

    print("=" * 96)
    print("最终普遍性: n2sjy2 纯迭代 80 轮 (无终止), a9=∠(v6,v) 历史最小")
    print("=" * 96)
    print(f"{'数据集':10s} {'a9初始':>8s} {'a9末轮':>8s} {'a9最小':>8s} {'≤10?':>6s} {'≤30?':>6s}")
    res = []
    for name, pts in datasets.items():
        try:
            f0, fl, fm = run_n2sjy2(name, pts)
            res.append((name, fm))
            print(f"{name:10s} {f0:8.1f} {fl:8.1f} {fm:8.1f} "
                  f"{'✅' if fm <= 10 else '✗':>6s} {'✅' if fm <= 30 else '✗':>6s}", flush=True)
        except Exception as e:
            print(f"{name:10s} 失败 {e}", flush=True)
    print()
    n10 = sum(1 for _, fm in res if fm <= 10)
    n30 = sum(1 for _, fm in res if fm <= 30)
    n50 = sum(1 for _, fm in res if fm <= 50)
    print(f"a9最小≤10°: {n10}/{len(res)} | ≤30°: {n30}/{len(res)} | ≤50°: {n50}/{len(res)}")
    print("DONE")


if __name__ == "__main__":
    main()
