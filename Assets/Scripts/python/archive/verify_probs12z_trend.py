# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
probs12z 趋势对比: probs12z(每轮当前t1,t2的格点概率场) 像 s0(PCA焦点场) 还是 s5(当前方向焦点场)?
  注意构造不同: probs12z 是格点距离 probTotal; s0/s5 是 BatchProbability(3D焦点对)
  比较: 每轮 corr(probs12z, s0) 与 corr(probs12z, s5) 随迭代的变化
"""
import math
import numpy as np

import data_driven_axis as dd
import n2sjy2 as n2
import nsjy_algorithms as m
from scipy.optimize import minimize


def angle(u, v):
    return math.degrees(math.acos(np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1)))


def bp(P, F1, F2, a):
    d1 = np.linalg.norm(P - F1, axis=1)
    d2 = np.linalg.norm(P - F2, axis=1)
    delta = d1 + d2 - 2 * a
    return np.exp(-np.abs(delta) / (2 * a))


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


def run_track(name, pts, max_outer=15):
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

    v1 = norm(F1 - F2)
    v3 = norm(F01 - F02)

    s0 = bp(P, v * c, -v * c, a)          # 固定 PCA 焦点场
    # 参考: 理论taus与(0,1)taus的格点场 (固定)
    pt_th, _, _, _ = dd.fast_probs(r30, r45, t1t, t2t, a)   # probs12[0]
    pt_01, _, _, _ = dd.fast_probs(r30, r45, t_, t_, a)     # probs012[0]

    vj = v + norm(F1 - F2)
    t1, t2 = t_, t_
    f1z, f2z = f1, f2
    print(f"  [{name}]")
    print(f"  {'iter':>4s} {'cond':>6s} | "
          f"{'corr(12z,s0)':>12s} {'corr(12z,s5)':>12s} {'corr(12z,12)':>12s} {'corr(12z,012)':>13s} | "
          f"{'corr(s0,s5)':>11s}")
    for i in range(max_outer):
        d = norm(f1z - f2z)
        ad1 = angle(d, v1)
        ad2 = angle(d, v3)
        cond = abs((ad2 + ad1) * 0.5 - min(ad2, ad1))
        pt_cur, _, _, _ = dd.fast_probs(r30, r45, t1, t2, a)   # probs12z
        s5 = bp(P, d * c, -d * c, a)
        c12z_s0 = np.corrcoef(pt_cur, s0)[0, 1]
        c12z_s5 = np.corrcoef(pt_cur, s5)[0, 1]
        c12z_12 = np.corrcoef(pt_cur, pt_th)[0, 1]
        c12z_012 = np.corrcoef(pt_cur, pt_01)[0, 1]
        cs0s5 = np.corrcoef(s0, s5)[0, 1]
        print(f"  {i:4d} {cond:6.2f} | {c12z_s0:12.3f} {c12z_s5:12.3f} "
              f"{c12z_12:12.3f} {c12z_012:13.3f} | {cs0s5:11.3f}", flush=True)
        if cond <= 8:
            print(f"    ==> 达成 iter{i} cond={cond:.2f}")
            break
        if i >= max_outer - 1:
            break
        # NM 一步
        def nm_obj(x, pcav, f1f2v):
            tt1 = complex(x[0], x[1])
            tt2 = complex(x[2], x[3])
            ptc, pc1, pc2, sc = dd.fast_probs(r30, r45, tt1, tt2, a)
            Fc1, Fc2 = m.extract_foci(pts, pc1, sc[1], pc2, sc[2])
            fc1, fc2 = fit_fast(P, ptc, Fc1, Fc2, a)
            dd_ = fc1 - fc2
            if np.dot(dd_, dd_) < 1e-12:
                return 0.
            dd_ = dd_ / np.linalg.norm(dd_)
            return abs(angle(dd_, pcav) - angle(dd_, f1f2v))

        lb = np.array([-0.5, 0.5, -0.5, 0.5])
        ub = np.array([0.5, 1.5, 0.5, 1.5])
        axis = np.cross(norm(F1 - F2), norm(F1 - F2)) * 0 + np.cross(v1, v3) * 0  # 占位
        # 简化旋转 (与先前一致)
        ax = np.cross(norm(f1z - f2z), v1)
        if np.linalg.norm(ax) < 1e-9:
            ax = np.array([1., 0, 0])
        def rotate(vv, a_, deg):
            if np.linalg.norm(a_) < 1e-12:
                return vv
            a_ = a_ / np.linalg.norm(a_)
            rg = math.radians(deg)
            return vv * math.cos(rg) + np.cross(a_, vv) * math.sin(rg) \
                + a_ * np.dot(a_, vv) * (1 - math.cos(rg))
        newv = rotate(v3, np.cross(norm(f1z - f2z), v1), -ad2)
        newF = rotate(v1, np.cross(norm(f1z - f2z), v3), -ad2)
        def cb(xk):
            return nm_obj(xk, newv, newF) <= 10.
        x0 = np.clip([t1.real, t1.imag, t2.real, t2.imag], lb, ub)
        try:
            res = minimize(nm_obj, x0, args=(newv, newF), method="Nelder-Mead",
                           bounds=list(zip(lb, ub)), callback=cb,
                           options={"maxiter": 400, "maxfev": 800, "xatol": 1e-6, "fatol": 1e-6})
            t1 = complex(res.x[0], res.x[1])
            t2 = complex(res.x[2], res.x[3])
        except Exception:
            pass
        ptc, pc1, pc2, sc = dd.fast_probs(r30, r45, t1, t2, a)
        Fc1, Fc2 = m.extract_foci(pts, pc1, sc[1], pc2, sc[2])
        f1z, f2z = fit_fast(P, ptc, Fc1, Fc2, a)


def main():
    print("=" * 110)
    print("probs12z 趋势: 像 s0(PCA场) 还是 s5(当前方向场)?  (corr 逐轮)")
    print("=" * 110)
    for name, pts in [("ball0", dd.make_ball(0)), ("ball2", dd.make_ball(2)),
                      ("ellip0", dd.make_ellip(0))]:
        try:
            run_track(name, pts)
        except Exception as e:
            print(f"  {name}: 失败 ({e})", flush=True)
        print(flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
