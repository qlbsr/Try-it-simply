# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
忠实运行 Python 原始 n2sjy2.py 主流程 (__main__ 部分) 在 pyjson 上
  对比: 角度趋势 (angle_pca=∠(dir,v_pca), angle1=∠(dir,init_dir)) 是否与 C# 版一致
  Python 原始版每轮: refine_moduli_by_axis(max_iter=50, axis=当前dir) + 重拟合
  C# 版有旋转+NM (Python 原始无)
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


def refine_lm(pts, r30, r45, a, t10, t20, axis_v, n_iter=50, track=False):
    P = np.array(pts, float)
    axis = np.asarray(axis_v, float)
    axis = axis / np.linalg.norm(axis)
    x = np.array([t10.real, t10.imag, t20.real, t20.imag], float)
    ang_hist = []

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
        r[len(pts)] = 20.0 * e[0]
        r[len(pts) + 1] = 20.0 * e[1]
        r[len(pts) + 2] = 20.0 * e[2]
        return r, F1c, F2c

    lam = 1e-3
    for it in range(n_iter):
        r, F1, F2 = resid(x)
        cost = float(np.sum(r * r))
        dv = F1 - F2
        ang = 180.0
        if np.linalg.norm(dv) > 1e-12:
            dv = dv / np.linalg.norm(dv)
            if np.dot(dv, axis) < 0:
                dv = -dv
            ang = angle(dv, axis)
        if track:
            ang_hist.append(ang)
        if ang < 0.5:
            break
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
    return complex(x[0], x[1]), complex(x[2], x[3]), F1, F2, ang_hist


def run_py_original(max_outer=60):
    """忠实 Python n2sjy2.py __main__ (用 pyjson)"""
    with open(PYJSON, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # Python 原始: np.random.seed(10); uniform(-1,1,(200,3)) — 但用 pyjson 对比
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
    F1_init, F2_init = m.extract_foci(pts, p1, sig[1], p2, sig[2])
    F1_fit, F2_fit = fit_fast(P, pt, F1_init, F2_init, a)
    f1, f2 = F1_fit, F2_fit
    init_dir = (f1 - f2) / np.linalg.norm(f1 - f2)
    v_pca = m.pca(pts)[0]
    v_pca = v_pca / np.linalg.norm(v_pca)
    # 真主轴(对照)
    C = P.T @ P
    evals, evecs = np.linalg.eigh(C)
    v_true = evecs[:, np.argsort(evals)[::-1][0]]
    t1, t2 = t_, t_
    print("Python 原始版 (pyjson, 无旋转/NM):")
    print(f"  {'iter':>4s} {'angle1(对init)':>13s} {'angle_pca(v3)':>13s} {'对真主轴':>8s} {'first_ang':>9s}")
    rows = []
    for i in range(max_outer):
        ptn, p1n, p2n, sgn = dd.fast_probs(r30, r45, t1, t2, a)
        F1n, F2n = m.extract_foci(pts, p1n, sgn[1], p2n, sgn[2])
        # 内层: refine_moduli_by_axis(axis=当前dir, max_iter=50)
        dir_cur = (f1 - f2) / np.linalg.norm(f1 - f2)
        t1n, t2n, F1n2, F2n2, ahist = refine_lm(
            pts, r30, r45, a, complex(0, 1), complex(0, 1), dir_cur,
            n_iter=50, track=True)
        # 重拟合焦点 (Python 原始第452行)
        ptn2, pn1, pn2, sgn2 = dd.fast_probs(r30, r45, t1n, t2n, a)
        f1n, f2n = fit_fast(P, ptn2, F1n2, F2n2, a)
        # 角度
        dir_vec = (f1n - f2n) / np.linalg.norm(f1n - f2n)
        angle1 = angle(dir_vec, init_dir)
        angle_pca = angle(dir_vec, v_pca)
        angle_true = angle(dir_vec, v_true)
        fa = ahist[0] if ahist else float('nan')
        rows.append((angle1, angle_pca, angle_true, fa))
        if i % 4 == 0 or i < 6:
            print(f"  {i:4d} {angle1:13.1f} {angle_pca:13.1f} {angle_true:8.1f} {fa:9.1f}",
                  flush=True)
        # 更新
        t1, t2 = t1n, t2n
        f1, f2 = f1n, f2n
    return rows


def main():
    rows = run_py_original()
    print()
    a1 = [r[0] for r in rows]
    ap = [r[1] for r in rows]
    at = [r[2] for r in rows]
    print(f"趋势: angle1 {a1[0]:.0f}→{a1[-1]:.0f} | angle_pca {ap[0]:.0f}→{ap[-1]:.0f} "
          f"| 对真主轴 {at[0]:.0f}→{at[-1]:.0f}")
    # 与 C# 版对比: C# pyjson 角度 (之前记录)
    print()
    print("对比: C# 版 pyjson 趋势 (阶段AD记录): angNew 30→38→76→109 (爬升), "
          "a9 降/卡 | Python 原始版见上")
    print("DONE")


if __name__ == "__main__":
    main()
