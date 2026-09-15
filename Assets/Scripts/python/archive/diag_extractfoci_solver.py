# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
诊断: pyjson 上 Unity vs Python 分歧根因候选
  A. lattice 概率场一致性 (真taus)
  B. ExtractFoci: MathNet AtA.Solve(正规方程) vs np.linalg.lstsq(SVD) — 病态时差异大
  C. FitFociByProbability(lr=0.001) 起点不同 → 终点方向
对照 Unity iter0: angleDegnew(∠f1-f2,f01-f02)=111.2, 内层首angle(∠ExtractF1-F2, axis)=70.7
"""
import json
import math

import numpy as np

import n2sjy2 as n2
import nsjy_algorithms as m


def angle(u, vv):
    return math.degrees(math.acos(np.clip(np.dot(u, vv) / (np.linalg.norm(u) * np.linalg.norm(vv)), -1, 1)))


def norm(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def fit_foci_cs(P, prob, F1, F2, a, iters=300, lr=0.001):
    F1 = np.asarray(F1, float).copy()
    F2 = np.asarray(F2, float).copy()
    for _ in range(iters):
        d1 = np.linalg.norm(P - F1, axis=1)
        d2 = np.linalg.norm(P - F2, axis=1)
        delta = d1 + d2 - 2.0 * a
        fp = np.exp(-np.abs(delta) / (2.0 * a))
        diff = fp - prob
        loss = float(np.sum(diff * diff))
        if loss < 1e-12:
            break
        sign = np.where(delta >= 0, 1.0, -1.0)
        coef = diff * fp * sign / (2.0 * a)
        m1 = d1 > 1e-6
        m2 = d2 > 1e-6
        g1 = np.sum(coef[m1, None] * (P[m1] - F1) / d1[m1, None], axis=0) if m1.any() else np.zeros(3)
        g2 = np.sum(coef[m2, None] * (P[m2] - F2) / d2[m2, None], axis=0) if m2.any() else np.zeros(3)
        F1 -= lr * g1
        F2 -= lr * g2
    return F1, F2


def lattice_probs(r30, r45, t1, t2, a, rng=20):
    Z1 = np.asarray(r30, complex)
    Z2 = np.asarray(r45, complex)
    mg = np.arange(-rng, rng + 1)
    MM, NN = np.meshgrid(mg, mg)
    L1 = MM.ravel() + NN.ravel() * t1
    L2 = MM.ravel() + NN.ravel() * t2
    d1 = np.abs(Z1[:, None] - L1[None, :]).min(axis=1)
    d2 = np.abs(Z2[:, None] - L2[None, :]).min(axis=1)
    s1 = np.sqrt(np.mean(d1 ** 2))
    s2 = np.sqrt(np.mean(d2 ** 2))
    p1 = np.exp(-d1 ** 2 / (2 * s1 ** 2))
    p2 = np.exp(-d2 ** 2 / (2 * s2 ** 2))
    pt = np.exp(-np.abs(d1 + d2 - 2 * a) / (2 * a))
    return pt, p1, p2, np.array([2 * a, s1, s2])


def extract_foci_normal_eq(points, prob_focus1, sigma1, prob_focus2, sigma2):
    """C# FitFocus: 正规方程 AtA.Solve(Atb) (MathNet 稠密求解, 非 SVD)"""
    n = len(points)
    dist1 = [sigma1 * math.sqrt(-2.0 * math.log(min(max(p, 1e-6), 1.0 - 1e-6))) for p in prob_focus1]
    dist2 = [sigma2 * math.sqrt(-2.0 * math.log(min(max(p, 1e-6), 1.0 - 1e-6))) for p in prob_focus2]
    return fit_focus_normal_eq(points, dist1), fit_focus_normal_eq(points, dist2)


def fit_focus_normal_eq(points, distances):
    """C# FitFocus (n2sjy2.cs 514-543): A = 2*(p_i - p0), AtA.Solve(Atb)"""
    p0 = np.asarray(points[0], float)
    d0 = distances[0]
    A = np.zeros((len(points) - 1, 3))
    b = np.zeros(len(points) - 1)
    for i in range(1, len(points)):
        diff = np.asarray(points[i], float) - p0
        A[i - 1] = 2.0 * diff
        b[i - 1] = np.dot(points[i], points[i]) - np.dot(p0, p0) \
                   - (distances[i] ** 2 - d0 ** 2)
    AtA = A.T @ A
    Atb = A.T @ b
    try:
        return np.linalg.solve(AtA, Atb)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]


def main():
    with open(r"C:\Users\23128\My project (2)\Assets\Resources\pyjson.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    pts = [np.array([p["x"], p["y"], p["z"]], float) for p in raw]
    P = np.array(pts, float)
    rp = n2.compute_rp(pts)
    d2 = math.asin(math.cos(math.radians(30)) / math.pi)
    a = rp * (1 + math.sin(d2))
    _, r45, r30, _ = n2.yzqx(pts, rp)
    t1t, t2t = n2.compute_taus()
    t_ = complex(0, 1)
    v3 = norm(m.pca(pts)[1])

    pt12, p112, p212, sig12 = lattice_probs(r30, r45, t1t, t2t, a)
    pt01, p101, p201, sig01 = lattice_probs(r30, r45, t_, t_, a)

    print("=== B. ExtractFoci: 正规方程 vs lstsq ===")
    F1n, F2n = extract_foci_normal_eq(pts, p112, sig12[1], p212, sig12[2])
    F1s, F2s = m.extract_foci(pts, p112, sig12[1], p212, sig12[2])
    F01n, F02n = extract_foci_normal_eq(pts, p101, sig01[1], p201, sig01[2])
    F01s, F02s = m.extract_foci(pts, p101, sig01[1], p201, sig01[2])
    print(f"真taus: F1 正规={F1n}  lstsq={F1s}  差={np.linalg.norm(F1n-F1s):.3e}")
    print(f"真taus: F2 正规={F2n}  lstsq={F2s}  差={np.linalg.norm(F2n-F2s):.3e}")
    print(f"(0,1):  F01 正规={F01n}  lstsq={F01s}  差={np.linalg.norm(F01n-F01s):.3e}")
    print(f"(0,1):  F02 正规={F02n}  lstsq={F02s}  差={np.linalg.norm(F02n-F02s):.3e}")

    print("\n=== AtA 条件数 (真taus prob1 的 A) ===")
    p0 = P[0]
    d0_ = sig12[1] * math.sqrt(-2.0 * math.log(min(max(p112[0], 1e-6), 1 - 1e-6)))
    Aa = np.zeros((len(pts) - 1, 3))
    for i in range(1, len(pts)):
        Aa[i - 1] = 2.0 * (P[i] - p0)
    print(f"cond(AtA) = {np.linalg.cond(Aa.T @ Aa):.3e}")

    print("\n=== C. fit(lr=0.001, 300步) 从两种起点 ===")
    for tag, F1, F2 in [("正规方程", F1n, F2n), ("lstsq", F1s, F2s)]:
        f1, f2 = fit_foci_cs(P, pt12, F1, F2, a)
        print(f"[真taus {tag}] f1-f2 dir vs v3 = {angle(norm(f1-f2), v3):.2f}°")
    for tag, F1, F2 in [("正规方程", F01n, F02n), ("lstsq", F01s, F02s)]:
        f01, f02 = fit_foci_cs(P, pt01, F01, F02, a)
        # 与 lstsq 真taus 的 f1/f2 比 (需要一致配对)
        pass

    # 配对组合计算 angleDegnew
    print("\n=== angleDegnew = ∠(f1-f2, f01-f02) 四种组合 ===")
    f1_NE, f2_NE = fit_foci_cs(P, pt12, F1n, F2n, a)
    f1_LS, f2_LS = fit_foci_cs(P, pt12, F1s, F2s, a)
    fo1_NE, fo2_NE = fit_foci_cs(P, pt01, F01n, F02n, a)
    fo1_LS, fo2_LS = fit_foci_cs(P, pt01, F01s, F02s, a)
    combos = {
        "NE/NE (C#完全对齐)": (f1_NE, f2_NE, fo1_NE, fo2_NE),
        "LS/LS (Python默认)": (f1_LS, f2_LS, fo1_LS, fo2_LS),
        "NE真/LS假": (f1_NE, f2_NE, fo1_LS, fo2_LS),
        "LS真/NE假": (f1_LS, f2_LS, fo1_NE, fo2_NE),
    }
    for tag, (f1, f2, fo1, fo2) in combos.items():
        an = angle(norm(f1 - f2), norm(fo1 - fo2))
        an_v3 = angle(norm(f1 - f2), v3)
        print(f"  {tag}: angleDegnew={an:.3f}°  (Unity 111.2141) | ∠(f1-f2,v3)={an_v3:.2f}°")


if __name__ == "__main__":
    main()
