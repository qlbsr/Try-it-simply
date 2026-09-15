# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
核心命题验证: 角平分线 bis = norm(v1+v3) 是否恒比 v1、v3 更接近 v(PCA)?
  v1 = (F1-F2) 方向, v3 = (F01-F02) 方向, v = PCA 主轴(真协方差主轴)
  测: a=∠(v1,v), b=∠(v3,v), c=∠(bis,v)
  检查: c < min(a,b) 是否恒成立? 成立率?
"""
import json
import math
import numpy as np

import data_driven_axis as dd
import n2sjy2 as n2
import nsjy_algorithms as m

PYJSON = r"C:\Users\23128\My project (2)\Assets\Resources\pyjson.json"


def angle(u, vv):
    return math.degrees(math.acos(np.clip(np.dot(u, vv) / (np.linalg.norm(u) * np.linalg.norm(vv)), -1, 1)))


def fit_fast(P, prob, F1, F2, a, iters=60, lr=0.0001):
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


def check(name, pts):
    pts = [np.array(p, float) for p in pts]
    P = np.array(pts, float)
    rp = n2.compute_rp(pts)
    d2 = math.asin(math.cos(math.radians(30)) / math.pi)
    a0 = rp * (1 + math.sin(d2))
    e3 = math.cos(d2) * 2 / (1 + math.sin(d2))
    c = rp * math.cos(d2) * e3
    _, r45, r30, _ = n2.yzqx(pts, rp)
    t1t, t2t = n2.compute_taus()
    t_ = complex(0, 1)
    pt, p1, p2, sig = dd.fast_probs(r30, r45, t1t, t2t, a0)
    F1, F2 = m.extract_foci(pts, p1, sig[1], p2, sig[2])
    f1, f2 = fit_fast(P, pt, F1, F2, a0)
    pt01, p101, p201, sig01 = dd.fast_probs(r30, r45, t_, t_, a0)
    F01, F02 = m.extract_foci(pts, p101, sig01[1], p201, sig01[2])
    f01, f02 = fit_fast(P, pt01, F01, F02, a0)
    # 真协方差主轴 (v)
    C = P.T @ P
    evals, evecs = np.linalg.eigh(C)
    v = evecs[:, np.argsort(evals)[::-1][0]]

    def norm(vv):
        return np.asarray(vv, float) / np.linalg.norm(vv)

    v1 = norm(F1 - F2)          # 理论提取 (F1-F2)
    v3 = norm(F01 - F02)        # (0,1) 提取 (F01-F02)
    bis_F = norm(v1 + v3)       # 提取焦点方向的角平分线
    # 拟合版
    v1f = norm(f1 - f2)
    v3f = norm(f01 - f02)
    bis_f = norm(v1f + v3f)

    a1 = angle(v1, v)
    a3 = angle(v3, v)
    cF = angle(bis_F, v)
    a1f = angle(v1f, v)
    a3f = angle(v3f, v)
    cf = angle(bis_f, v)
    # 角平分线 vs 最小边
    ok_F = cF < min(a1, a3) - 1e-6
    ok_f = cf < min(a1f, a3f) - 1e-6
    print(f"  {name:10s}: ∠(v1,v)={a1:5.1f} ∠(v3,v)={a3:5.1f} "
          f"∠(bisF,v)={cF:5.1f} {'✅' if ok_F else '✗'} | "
          f"拟合: ∠(f1,v)={a1f:5.1f} ∠(f3,v)={a3f:5.1f} ∠(bisf,v)={cf:5.1f} "
          f"{'✅' if ok_f else '✗'}")
    return ok_F, ok_f


def main():
    dats = {}
    with open(PYJSON, "r", encoding="utf-8") as f:
        raw = json.load(f)
    dats["pyjson"] = [np.array([p["x"], p["y"], p["z"]], float) for p in raw]
    for seed in range(4):
        dats[f"ball{seed}"] = dd.make_ball(seed)
    for seed in range(4):
        dats[f"ellip{seed}"] = dd.make_ellip(seed)
    for seed in range(3):
        dats[f"cube{seed}"] = dd.make_cube(10 + seed)
    print("=" * 110)
    print("角平分线命题: ∠(bis,v) < min(∠(v1,v),∠(v3,v)) 恒成立?")
    print("=" * 110)
    okF = okf = 0
    n = 0
    for name, pts in dats.items():
        try:
            r1, r2 = check(name, pts)
            okF += r1
            okf += r2
            n += 1
        except Exception as e:
            print(f"  {name}: 失败 {e}", flush=True)
    print()
    print(f"提取焦点角平分线成立: {okF}/{n} | 拟合焦点角平分线成立: {okf}/{n}")
    print("DONE")


if __name__ == "__main__":
    main()
