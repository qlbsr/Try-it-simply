# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
对比两套 compute_taus + 各自驱动的初始拟合, 对照 Unity iter0:
  new = ∠(f1-f2, f01-f02) = 111.2   (line 85, f1/f2=真taus拟合, f01/f02=(0,1)拟合)
  odr = RefineModuliByAxis 首内迭代 ∠(ExtractF1-F2@(0,1), axi=d_cur) = 70.7
"""
import json
import math

import numpy as np

import n2sjy2 as n2
import nsjy_algorithms as m
import data_driven_axis as dd

# n2sjy2 的 mpmath 版已被 data_driven_axis monkey-patch compute_probabilities... 无碍


def norm(v):
    return np.asarray(v, float) / np.linalg.norm(v)


def angle(u, vv):
    return math.degrees(math.acos(np.clip(np.dot(u, vv) / (np.linalg.norm(u) * np.linalg.norm(vv)), -1, 1)))


def fit_foci_n2(P, prob, F1, F2, a, iters=300, lr=0.0001, lambda_sep=1.0, eps=1e-3):
    F1 = np.asarray(F1, float).copy()
    F2 = np.asarray(F2, float).copy()
    for _ in range(iters):
        d1 = np.linalg.norm(P - F1, axis=1)
        d2 = np.linalg.norm(P - F2, axis=1)
        sd1 = np.maximum(d1, eps)
        sd2 = np.maximum(d2, eps)
        delta = d1 + d2 - 2.0 * a
        fp = np.exp(-np.abs(delta) / (2.0 * a))
        diff = fp - prob
        loss = float(np.sum(diff * diff)) + lambda_sep * float(np.dot(F1 - F2, F1 - F2))
        if loss < 1e-12:
            break
        sign = np.where(delta >= 0, 1.0, -1.0)
        coef = diff * fp * sign / (2.0 * a)
        g_raw1 = coef[:, None] * (P - F1) / sd1[:, None]
        g_raw2 = coef[:, None] * (P - F2) / sd2[:, None]
        n1 = np.linalg.norm(g_raw1, axis=1)
        n2 = np.linalg.norm(g_raw2, axis=1)
        s1 = np.where(n1 > 1.0, 1.0 / n1, 1.0)
        s2 = np.where(n2 > 1.0, 1.0 / n2, 1.0)
        grad1 = np.sum(g_raw1 * s1[:, None], axis=0)
        grad2 = np.sum(g_raw2 * s2[:, None], axis=0)
        sep = F1 - F2
        grad1 += 2.0 * lambda_sep * sep
        grad2 -= 2.0 * lambda_sep * sep
        gn1 = np.linalg.norm(grad1)
        gn2 = np.linalg.norm(grad2)
        if gn1 > 5.0:
            grad1 *= 5.0 / gn1
        if gn2 > 5.0:
            grad2 *= 5.0 / gn2
        F1 -= lr * grad1
        F2 -= lr * grad2
        if not (np.all(np.isfinite(F1)) and np.all(np.isfinite(F2))):
            break
    return F1, F2


def main():
    with open(r"C:\Users\23128\My project (2)\Assets\Resources\pyjson.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    pts = [np.array([p["x"], p["y"], p["z"]], float) for p in raw]
    P = np.array(pts, float)
    rp = n2.compute_rp(pts)
    d2 = math.asin(math.cos(math.radians(30)) / math.pi)
    a = rp * (1 + math.sin(d2))
    _, r45, r30, _ = n2.yzqx(pts, rp)
    t_ = complex(0, 1)
    v3 = norm(m.pca(pts)[1])

    tau_sets = {
        "n2sjy2(mpmath)": n2.compute_taus(),
        "nsjy_algo(Carlsonfk)": m.compute_taus(),
    }
    # (0,1) 概率场 (两套 tau 无关, 只算一次)
    pt01, p101, p201, sig01 = dd.fast_probs(r30, r45, t_, t_, a)
    F01, F02 = m.extract_foci(pts, p101, sig01[1], p201, sig01[2])
    f01, f02 = fit_foci_n2(P, pt01, F01, F02, a)
    d01 = norm(f01 - f02)
    print(f"(0,1) 拟合: f01-f02 vs v3 = {angle(d01, v3):.2f}°")

    for tag, (t1t, t2t) in tau_sets.items():
        print(f"\n===== {tag}: t1={t1t} t2={t2t} =====")
        pt12, p112, p212, sig12 = dd.fast_probs(r30, r45, t1t, t2t, a)
        F1, F2 = m.extract_foci(pts, p112, sig12[1], p212, sig12[2])
        f1, f2 = fit_foci_n2(P, pt12, F1, F2, a)
        d12 = norm(f1 - f2)
        print(f"  真taus拟合: f1={f1}")
        print(f"             f2={f2}")
        print(f"  |f1-f2|={np.linalg.norm(f1-f2):.4f}")
        print(f"  f1-f2 vs v3    = {angle(d12, v3):.2f}°")
        print(f"  f1-f2 vs (0,1)d = {angle(d12, d01):.2f}°   ← angleDegnew 期望 111.2")
        # RefineModuliByAxis 首内迭代: x=(0,1),(0,1), axis=d12
        axis = d12
        x = np.array([0.0, 1.0, 0.0, 1.0])
        ptc, pc1, pc2, sc = dd.fast_probs(r30, r45, complex(x[0], x[1]), complex(x[2], x[3]), a)
        Fc1, Fc2 = m.extract_foci(pts, pc1, sc[1], pc2, sc[2])
        dE = norm(Fc1 - Fc2)
        if np.dot(dE, axis) < 0:
            dE = -dE
        print(f"  ExtractF1-F2@(0,1) vs v3 = {angle(Fc1-Fc2, v3):.2f}°")
        print(f"  首内迭代 angle ∠(ExtractF1-F2, axis) = {angle(dE, axis):.2f}°   ← odr 期望 70.7")


if __name__ == "__main__":
    main()
