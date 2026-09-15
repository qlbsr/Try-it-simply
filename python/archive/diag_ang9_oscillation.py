# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
诊断: angleDeg9=∠(v6, v) 无终止时逐渐减小但震荡回去
  问题: ① 是否单调下降趋势? ② 震荡幅度/周期? ③ 震荡源: NM步长过大? 旋转符号? 参考跳变?
  追踪: 每轮 v6 方向 + angleDeg9 + t1/t2 移动量 + NM 求值次数
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


def run_diag(name, pts, max_outer=30):
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
    vj = v + norm(F1 - F2)
    t1, t2 = t_, t_
    f1z, f2z = f1, f2
    print(f"  [{name}] ∠(v2,v)={angle(v2, v):.1f}°  ∠(v4,v)={angle(v4, v):.1f}°")
    print(f"  {'iter':>4s} {'ang9(∠v6,v)':>12s} {'Δang9':>7s} {'ang5':>6s} {'ang6':>6s} "
          f"{'|Δt1|':>7s} {'|Δt2|':>7s} {'evals':>6s}")
    prev9 = None
    for i in range(max_outer):
        # LM 精化 (模拟 RefineModuliByAxis) — 简化: 直接用当前t提取
        t1n, t2n = t1, t2
        ptn, p1n, p2n, sgn = dd.fast_probs(r30, r45, t1n, t2n, a)
        F1n, F2n = m.extract_foci(pts, p1n, sgn[1], p2n, sgn[2])
        f1n, f2n = fit_fast(P, ptn, F1n, F2n, a)
        v6 = norm(f1n - f2n)
        ang9 = angle(v6, v)
        ang5 = angle(v6, v2)
        ang6 = angle(v6, v4)
        d9 = ang9 - prev9 if prev9 is not None else float('nan')
        prev9 = ang9
        if i % 2 == 0 or (i < 12):
            print(f"  {i:4d} {ang9:12.1f} {d9:7.1f} {ang5:6.1f} {ang6:6.1f} "
                  f"{abs(t1-t1n):7.3f} {abs(t2-t2n):7.3f}", flush=True)
        # NM 精化 (RefineTausWithNM 的等价: 最小化 |∠(d,newf)-∠(d,newv)|)
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
        ub = np.array([0.5, 2.0, 0.5, 2.0])
        # 用户旋转逻辑: 旋转 v2/v4 到 v6
        axis = np.cross(v6, v2)
        axis1 = np.cross(v6, v4)

        def rotate(vv, a_, deg):
            if np.linalg.norm(a_) < 1e-12:
                return vv
            a_ = a_ / np.linalg.norm(a_)
            rg = math.radians(deg)
            return vv * math.cos(rg) + np.cross(a_, vv) * math.sin(rg) \
                + a_ * np.dot(a_, vv) * (1 - math.cos(rg))

        newf = rotate(v2, axis, -ang5)
        newv = rotate(v4, axis1, -ang6)

        def cb(xk):
            return nm_obj(xk, newf, newv) <= 10.

        x0 = np.clip([t1.real, t1.imag, t2.real, t2.imag], lb, ub)
        ev0 = [0]

        def nm_obj2(x):
            ev0[0] += 1
            return nm_obj(x, newf, newv)

        try:
            rr = minimize(nm_obj2, x0, method="Nelder-Mead", bounds=list(zip(lb, ub)),
                          callback=cb, options={"maxiter": 600, "maxfev": 1200,
                                                "xatol": 1e-6, "fatol": 1e-6})
            t1 = complex(rr.x[0], rr.x[1])
            t2 = complex(rr.x[2], rr.x[3])
        except Exception:
            pass
    print()


def main():
    print("=" * 100)
    print("诊断: angleDeg9 无终止迭代轨迹 (每 2 轮打印)")
    print("=" * 100)
    for name, pts in [("ball1", dd.make_ball(1)), ("ellip0", dd.make_ellip(0)),
                      ("ball2", dd.make_ball(2))]:
        try:
            run_diag(name, pts)
        except Exception as e:
            print(f"  {name}: 失败 ({e})", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
