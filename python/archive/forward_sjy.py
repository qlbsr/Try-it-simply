# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
sjy 正向链 M1 (骨架版): v3=(d2,fj) → 概率场 → 复平面 → 复梯度 → ABCD → τ → j
忠实点:
  - ComputeProbabilities: h=rp cos d2, r=rp sin d2, a=(rp+r)/2, c=h*e3, F1,F2=±c*v3
  - BatchProbability C# 原始优先级: exp(-delta^2/8*a*a) = exp(-delta^2 * a^2/8)
  - InverseTh4: Rs=(|p|/rp)^(1/sin d2), theta=phi/sin d2
  - NumericalGradient: k=12 近邻局部线性拟合 (uvx,uvy,1) → (dw/du, dw/dv)
  - 复梯度 ∂/∂z = 0.5(dw/du - i dw/dv)
  - FitPolynomial: 基 [z, z̄, z², z̄²] 复最小二乘 → (A,B,C,D)
  - disc=B²-4AC; ratio=(B+√disc)/(B-√disc); |ratio|>1→1/ratio
  - τ=log(ratio)/(2πi); j=1/q+744+196884q+21493760q²+864299970q³
简化点 (待 M2 补): 叶映射几何(MapDoubleConeToLeaf)、S³ 球面 RBF、四元数项
"""
import json
import math
import sys

import numpy as np


def inverse_th4(p, d2_deg, rp):
    d2 = math.radians(d2_deg)
    x, y, z = p
    rs = math.hypot(x, y)
    zs = z
    phis = math.atan2(y, x)
    if phis < 0:
        phis += 2 * math.pi
    r = math.hypot(rs, zs)
    sin_d2 = math.sin(d2)
    thetas = phis / sin_d2
    thetas = thetas % (2 * math.pi)
    Rs = (r / rp) ** (1.0 / sin_d2)
    return complex(Rs * math.cos(thetas), Rs * math.sin(thetas))


def v3_from_angles(d2_deg, fj_deg):
    d2 = math.radians(d2_deg)
    fj = math.radians(fj_deg)
    return np.array([math.sin(d2) * math.cos(fj), math.cos(d2), math.sin(d2) * math.sin(fj)])


def compute_probabilities(P, v3, rp, d2_deg, e3=None):
    """C# sjy.ComputeProbabilities + BatchProbability (原始优先级 bug 版)"""
    d2 = math.radians(d2_deg)
    h = rp * math.cos(d2)
    r = rp * math.sin(d2)
    a = (rp + r) * 0.5
    if e3 is None:
        e3 = 2.0 * math.cos(d2) / (1.0 + math.sin(d2))   # 离心率式 (nsjy/n2sjy2/xinsjy 版)
    c = h * e3
    F1 = c * v3
    F2 = -c * v3
    d1 = np.linalg.norm(P - F1, axis=1)
    d2d = np.linalg.norm(P - F2, axis=1)
    delta = d1 + d2d - 2.0 * a
    # C#: Mathf.Exp(-Mathf.Pow(delta,2)/8f * a * a)  →  exp(-delta^2 * a^2 / 8)
    prob = np.exp(-(delta ** 2) * (a ** 2) / 8.0)
    return prob, dict(h=h, r=r, a=a, c=c, F1=F1, F2=F2, e3=e3)


def numerical_gradient(uv, values, K=12):
    """C# NumericalGradient: 每点取 K 近邻 (uvx,uvy,1) 线性拟合 → (dw/du, dw/dv)"""
    z = np.asarray(uv, complex)
    uvx = z.real
    uvy = z.imag
    vals = np.asarray(values, float)
    N = len(z)
    grad = np.zeros((N, 2))
    pts = np.stack([uvx, uvy], axis=1)
    for i in range(N):
        d = np.linalg.norm(pts - pts[i], axis=1)
        order = np.argsort(d)
        nbr = [j for j in order if d[j] > 1e-8][:K]
        if len(nbr) < 3:
            continue
        A = np.stack([uvx[nbr], uvy[nbr], np.ones(len(nbr))], axis=1)
        b = vals[nbr]
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        grad[i] = (sol[0], sol[1])
    return grad


def fit_polynomial(f, F):
    """C# FitPolynomial: 基 [z, z̄, z², z̄²] SVD 最小二乘"""
    f = np.asarray(f, complex)
    F = np.asarray(F, complex)
    X = np.stack([f, np.conj(f), f * f, np.conj(f) ** 2], axis=1)
    beta, *_ = np.linalg.lstsq(X, F, rcond=None)
    return beta[0], beta[1], beta[2], beta[3]


def tau_from_abcd(A, B, C, D):
    disc = B * B - 4.0 * A * C
    sq = np.sqrt(disc + 0j)
    den = B - sq
    if abs(den) < 1e-300:
        return complex(np.nan, np.nan), complex(np.nan, np.nan), disc
    ratio = (B + sq) / den
    if abs(ratio) > 1.0:
        ratio = 1.0 / ratio
    if ratio == 0:
        return complex(np.nan, np.nan), ratio, disc
    tau = np.log(ratio) / (2.0 * np.pi * 1j)
    return tau, ratio, disc


def j_invariant(tau):
    q = np.exp(2 * np.pi * 1j * tau)
    if abs(q) < 1e-300:
        return complex(np.inf, np.inf)
    return 1.0 / q + 744 + 196884 * q + 21493760 * q ** 2 + 864299970 * q ** 3


def forward_m1(P, d2_deg, fj_deg, rp, e3=None, w_mode="prob"):
    """M1 正向链: v3 → j"""
    v3 = v3_from_angles(d2_deg, fj_deg)
    prob, info = compute_probabilities(P, v3, rp, d2_deg, e3=e3)
    uvs = np.array([inverse_th4(p, d2_deg, rp) for p in P])
    if w_mode == "prob":
        w = prob
    else:  # 外部势: uvs 方向与 v3 投影
        nrm = np.abs(uvs)
        nrm[nrm < 1e-12] = 1e-12
        w = (uvs / nrm).real * v3[0] + (uvs / nrm).imag * v3[2]
    g = numerical_gradient(uvs, w, 12)
    Fz = 0.5 * (g[:, 0] - 1j * g[:, 1])          # ∂/∂z = 0.5(dw/du - i dw/dv)
    total = 6.0 * Fz
    total = np.where(np.abs(total) > 70.0, 0.0 + 0.0j, total)   # com=70 替换 (简化: 置0)
    A, B, C, D = fit_polynomial(uvs, total)
    tau, ratio, disc = tau_from_abcd(A, B, C, D)
    j = j_invariant(tau)
    return dict(v3=v3, prob=prob, uvs=uvs, w=w, ABCD=(A, B, C, D),
                tau=tau, ratio=ratio, disc=disc, j=j, info=info)


def main():
    path = r"C:\Users\23128\My project (2)\Assets\Resources\points.json"
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    pts = [np.array(p, float) for p in raw]
    P = np.array(pts, float)
    rp = float(np.mean(np.linalg.norm(P, axis=1)))
    print(f"点集 {path} n={len(P)} rp={rp:.4f}")

    # 扫描 v3 = (d2, fj) → j
    print(f"\n{'d2':>6s} {'fj':>7s} {'|prob|max':>10s} {'a':>6s} {'c':>6s} "
          f"{'tau.re':>9s} {'tau.im':>9s} {'j.re':>13s} {'j.im':>13s}")
    rows = []
    for d2d in [10, 20, 30, 45, 60, 75, 85]:
        for fjd in [0, 45, 90, 135, 180, 225, 270, 315]:
            r = forward_m1(P, d2d, fjd, rp)
            j = r["j"]
            tau = r["tau"]
            print(f"{d2d:6d} {fjd:7d} {r['prob'].max():10.4f} {r['info']['a']:6.3f} "
                  f"{r['info']['c']:6.3f} {tau.real:9.4f} {tau.imag:9.4f} "
                  f"{j.real:13.4f} {j.imag:13.4f}")
            rows.append((d2d, fjd, j, tau))
    js = np.array([r[2] for r in rows])
    print(f"\nj 统计: 最大模 {np.abs(js).max():.3e}  最小模 {np.abs(js).min():.3e}  "
          f"唯一值比例 {len(set(np.round(js.real,6).astype(str)))/len(js):.2f}")
    print("→ 若 j 随 (d2,fj) 单调可分 → 逆映射可学; 若大量重复/NaN → 需可微精修或多起点")


if __name__ == "__main__":
    main()
