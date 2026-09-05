# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
观察: s0[0] 与 min(s2[0], s4[0], s5[0]) 的近似关系并拟合
  收集: 多数据集迭代全程的 s0,s1,s2,s3,s4,s5 第一个元素
  拟合: s0[0] ≈ a·min(s2,s4,s5)[0] + b ?  或 s0[0] ≈ min + 常数?
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


def collect(name, pts, max_outer=15):
    """返回 (s0_0列表, min_0列表, cond列表, 达成点值)"""
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
    v2 = norm(f1 - f2)
    v3 = norm(F01 - F02)
    v4 = norm(f01 - f02)
    s0 = bp(P, v * c, -v * c, a)[0]
    s2 = bp(P, v2 * c, -v2 * c, a)[0]
    s4 = bp(P, v4 * c, -v4 * c, a)[0]

    vj = v + norm(F1 - F2)
    t1, t2 = t_, t_
    f1z, f2z = f1, f2
    rows = []
    achieved = None
    for i in range(max_outer):
        d = norm(f1z - f2z)
        ad1 = angle(d, v1)
        ad2 = angle(d, v3)
        cond = abs((ad2 + ad1) * 0.5 - min(ad2, ad1))
        s5 = bp(P, d * c, -d * c, a)[0]
        mn = min(s2, s4, s5)
        rows.append((s0, mn, cond))
        if cond <= 8:
            achieved = (s0, mn, s2, s4, s5, cond)
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

        def rotate(vv, a_, deg):
            if np.linalg.norm(a_) < 1e-12:
                return vv
            a_ = a_ / np.linalg.norm(a_)
            rg = math.radians(deg)
            return vv * math.cos(rg) + np.cross(a_, vv) * math.sin(rg) \
                + a_ * np.dot(a_, vv) * (1 - math.cos(rg))

        newv = rotate(v3, np.cross(d, v1), -ad2)
        newF = rotate(v1, np.cross(d, v3), -ad2)

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
    return rows, achieved


def main():
    datasets = {}
    for seed in range(4):
        datasets[f"ball{seed}"] = dd.make_ball(seed)
    for seed in range(3):
        datasets[f"ellip{seed}"] = dd.make_ellip(seed)
    datasets["cube0"] = dd.make_cube(10)
    datasets["cube2"] = dd.make_cube(12)

    print("=" * 96)
    print("s0[0] vs min(s2[0],s4[0],s5[0]) 全程收集")
    print("=" * 96)
    all_pairs = []
    for name, pts in datasets.items():
        try:
            rows, ach = collect(name, pts)
        except Exception as e:
            print(f"  {name}: 失败 ({e})", flush=True)
            continue
        for s0, mn, cond in rows:
            all_pairs.append((s0, mn))
        if ach:
            s0, mn, s2, s4, s5, cond = ach
            print(f"  {name}: 达成 cond={cond:.2f} | s0={s0*100:.2f} "
                  f"min(s2,s4,s5)={mn*100:.2f} (s2={s2*100:.2f} s4={s4*100:.2f} s5={s5*100:.2f})",
                  flush=True)
        else:
            print(f"  {name}: 未达成", flush=True)
    A = np.array(all_pairs) * 100   # ×100
    s0s = A[:, 0]
    mns = A[:, 1]
    print()
    print(f"共 {len(A)} 个样本点 (多数据集迭代全程)")
    print("=" * 96)
    print("拟合: s0[0] ≈ a·min + b")
    print("=" * 96)
    corr = np.corrcoef(s0s, mns)[0, 1]
    a_, b_ = np.polyfit(mns, s0s, 1)
    resid = s0s - (a_ * mns + b_)
    print(f"  corr(s0, min) = {corr:+.3f}")
    print(f"  s0 ≈ {a_:.3f}·min {b_:+.3f}   |残差|均值={np.abs(resid).mean():.3f} "
          f"P90={np.percentile(np.abs(resid),90):.3f}")
    # 检查 s0 与 min 谁大
    print(f"  s0>min 占比: {(s0s > mns).mean()*100:.1f}%  (s0<min: {(s0s < mns).mean()*100:.1f}%)")
    print(f"  s0-min 均值: {(s0s-mns).mean():.3f} 中位: {np.median(s0s-mns):.3f}")
    # 简单近似: s0 ≈ min + Δ
    print()
    print("备选拟合: s0 ≈ min + Δ (Δ 常数?)")
    print(f"  Δ=s0-min: 均值={np.mean(s0s-mns):.3f} std={np.std(s0s-mns):.3f}")
    # 散点摘要
    print()
    print("分箱摘要 (min 值箱 → s0 均值):")
    for lo in range(50, 90, 5):
        mk = (mns >= lo) & (mns < lo + 5)
        if mk.sum() >= 5:
            print(f"  min∈[{lo},{lo+5}): n={mk.sum():4d} s0均值={s0s[mk].mean():6.2f}")
    print("DONE")


if __name__ == "__main__":
    main()
