# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
快速测试 (忠实 n2sjy2.cs 当前循环, 简化为关键步骤):
  每轮: angleDegnew = ∠(d, f01-f02拟合)
        停滞检测: |angNew - prevAngNew| 连续 N 轮 < 1° → 停止 (角差无法下降)
  目标: 测试"角差无法下降时停止"对所有点集是否普遍触发且停止位置合理
  轻量: 外层≤40轮, LM 内层 20, 无 NM (NM 是次要的), fit 50
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


def fit_fast(P, prob, F1, F2, a, iters=50, lr=0.0001):
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


def refine_lm(pts, r30, r45, a, t10, t20, axis_v, n_iter=20):
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


def run_quick(name, pts, max_outer=40, stall_n=5):
    """停滞检测: |angNew变化|<1° 连续 stall_n 轮 → 停止"""
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
    t1, t2 = t_, t_
    f1z, f2z = f1, f2
    prev_new = None
    stall = 0
    stop_iter = None
    stop_ang = None
    best_a9v, best_a9t = 999.0, 999.0
    for i in range(max_outer):
        d_cur = norm(f1z - f2z)
        angNew = angle(f01fit, d_cur)
        # 停滞检测: angNew 无法再下降
        if prev_new is not None:
            if abs(angNew - prev_new) < 1.0:
                stall += 1
                if stall >= stall_n:
                    stop_iter = i
                    stop_ang = angNew
                    break
            else:
                stall = 0
        prev_new = angNew
        # 内层 LM (axis=d 自指, 忠实当前代码)
        t1n, t2n, F1n, F2n = refine_lm(pts, r30, r45, a, t_, t_, d_cur)
        ptn, p1n, p2n, sgn = dd.fast_probs(r30, r45, t1n, t2n, a)
        f1n, f2n = fit_fast(P, ptn, F1n, F2n, a)
        v6 = norm(f1n - f2n)
        a9v = angle(v6, v3)
        a9t = angle(v6, v_true)
        best_a9v = min(best_a9v, a9v)
        best_a9t = min(best_a9t, a9t)
        # 更新 t (简化: LM 结果即新 t; NM 省略以提速)
        t1, t2 = t1n, t2n
        f1z, f2z = f1n, f2n
    if stop_iter is None:
        stop_iter = max_outer
    return stop_iter, stop_ang, best_a9v, best_a9t


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
    dats["gauss5"] = [np.random.default_rng(1005).standard_normal(3) * (0.5 + 2 * np.random.default_rng(1006).random()) for _ in range(200)]

    print("=" * 100)
    print("快速测试: 停滞检测(angNew连续5轮变<1°)停止 | 全数据集")
    print("=" * 100)
    print(f"{'数据集':10s} {'停于iter':>8s} {'停时angNew':>10s} {'a9(v3)best':>10s} {'a9(真)best':>10s}")
    for name, pts in dats.items():
        try:
            si, sa, bv, bt = run_quick(name, pts)
            print(f"{name:10s} {si:8d} {sa if sa is not None else -1:10.1f} "
                  f"{bv:10.1f} {bt:10.1f}", flush=True)
        except Exception as e:
            print(f"{name:10s} 失败 {e}", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
