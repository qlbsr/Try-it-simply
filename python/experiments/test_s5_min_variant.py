# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
测试: #2 条件 + angleDeg1>=16 && angleDeg2>=16 + s5[0] 必须最小
  变体:
    A: cond=|(ang2+ang1)/2-min|<=8                       (原 #2)
    B: ang1>=16 && ang2>=16 && cond<=8
    C: B 且 s5[0] 是 s0..s5[0] 中的最小值
    D: B 且 s5[0] <= s2[0] 且 s5[0] <= s4[0]  (s5 最小 vs s2/s4)
  统计: 各变体的数据集达成率 / 达成迭代 / 达成时的质量
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


def run_variants(name, pts, max_outer=15):
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
    s1 = bp(P, v1 * c, -v1 * c, a)[0]
    s2 = bp(P, v2 * c, -v2 * c, a)[0]
    s3 = bp(P, v3 * c, -v3 * c, a)[0]
    s4 = bp(P, v4 * c, -v4 * c, a)[0]
    s0 = bp(P, v * c, -v * c, a)[0]

    vj = v + norm(F1 - F2)
    t1, t2 = t_, t_
    f1z, f2z = f1, f2
    # 每个变体记录: 是否达成, 达成迭代, 达成时 cond/质量
    res = {k: (False, None, None) for k in "ABCD"}
    for i in range(max_outer):
        d = norm(f1z - f2z)
        ad1 = angle(d, v1)
        ad2 = angle(d, v3)
        cond = abs((ad2 + ad1) * 0.5 - min(ad2, ad1))
        s5 = bp(P, d * c, -d * c, a)[0]
        # 约束
        ge16 = (ad1 >= 16) and (ad2 >= 16)
        s5_min_all = s5 <= min(s0, s1, s2, s3, s4)      # s5 是 s0..s4 中最小
        s5_min_24 = s5 <= min(s2, s4)                   # s5 <= s2 且 s5 <= s4
        variants = {
            "A": cond <= 8,
            "B": ge16 and cond <= 8,
            "C": ge16 and cond <= 8 and s5_min_all,
            "D": ge16 and cond <= 8 and s5_min_24,
        }
        for k in "ABCD":
            if variants[k] and res[k][0] is False:
                res[k] = (True, i, (ad1, ad2, cond, s0, s5, s2, s4))
        if all(res[k][0] for k in "ABCD"):
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
            rr = minimize(nm_obj, x0, args=(newv, newF), method="Nelder-Mead",
                          bounds=list(zip(lb, ub)), callback=cb,
                          options={"maxiter": 400, "maxfev": 800, "xatol": 1e-6, "fatol": 1e-6})
            t1 = complex(rr.x[0], rr.x[1])
            t2 = complex(rr.x[2], rr.x[3])
        except Exception:
            pass
        ptc, pc1, pc2, sc = dd.fast_probs(r30, r45, t1, t2, a)
        Fc1, Fc2 = m.extract_foci(pts, pc1, sc[1], pc2, sc[2])
        f1z, f2z = fit_fast(P, ptc, Fc1, Fc2, a)
    return res


def main():
    datasets = {}
    for seed in range(4):
        datasets[f"ball{seed}"] = dd.make_ball(seed)
    for seed in range(4):
        datasets[f"ellip{seed}"] = dd.make_ellip(seed)
    for seed in range(3):
        datasets[f"cube{seed}"] = dd.make_cube(10 + seed)
    import json
    with open(r"C:\Users\23128\My project (2)\Assets\Resources\pyjson.json", encoding="utf-8") as f:
        raw = json.load(f)
    datasets["pyjson"] = [np.array([p["x"], p["y"], p["z"]], float) for p in raw]

    print("=" * 110)
    print("变体测试: A=#2 | B=A+ang1,ang2>=16 | C=B+s5最小(全部) | D=B+s5<=min(s2,s4)")
    print("=" * 110)
    agg = {k: {"ok": 0, "iters": [], "cond_at": []} for k in "ABCD"}
    for name, pts in datasets.items():
        try:
            res = run_variants(name, pts)
        except Exception as e:
            print(f"  {name}: 失败 ({e})", flush=True)
            continue
        line = f"  {name:8s}: "
        for k in "ABCD":
            ok, it, info = res[k]
            if ok:
                agg[k]["ok"] += 1
                agg[k]["iters"].append(it)
                agg[k]["cond_at"].append(info[2])
                line += f"{k}=✅it{it} "
            else:
                line += f"{k}=✗ "
        print(line, flush=True)
    print()
    print("=" * 110)
    print("汇总")
    print("=" * 110)
    for k in "ABCD":
        a = agg[k]
        print(f"  {k}: 达成 {a['ok']}/{len(datasets)} ({a['ok']/len(datasets)*100:.0f}%)  "
              f"迭代均值 {np.mean(a['iters']) if a['iters'] else float('nan'):.1f}  "
              f"达成cond均值 {np.mean(a['cond_at']) if a['cond_at'] else float('nan'):.2f}")
    print("DONE")


if __name__ == "__main__":
    main()
