# -*- coding: utf-8 -*-
"""
验证: w 到底依赖哪些变量? (决定 RBF 能否被低维基替换)

方法: 核岭回归 α = (K+λI)^{-1}w, LOO 残差 = α_i / H_ii  (H = K(K+λI)^{-1})
      LOO_R2 = 1 - Σ(loo_i)² / Σ(w_i - mean w)²
      比较不同"特征集"作为核输入时的 LOO_R2:
        θ(环坐标) / ρ=|uv| / r=|P| / (θ,ρ) / (x,y,z) / (θ,ρ,高度) / uv(实虚)
结果判读:
  · 若 θ 单独就能达到很高 LOO_R2  → 一维周期展开可行 (路径 B 原方案)
  · 若必须用 3D 位置             → w 依赖完整位置, 必须用 X∈S³ 类核 (RBF 不可省)
"""
import json
import math

import numpy as np

from forward_sjy import compute_probabilities, v3_from_angles

PI = math.pi
RES = r"C:\Users\23128\My project (2)\Assets\Resources"


def load_points(fn):
    with open(f"{RES}\\{fn}", "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw[0], dict):
        return np.array([[p["x"], p["y"], p["z"]] for p in raw], float)
    return np.array([list(p) for p in raw], float)


def ring_theta(P, d2_deg):
    phis = np.arctan2(P[:, 1], P[:, 0])
    phis = np.where(phis < 0, phis + 2 * PI, phis)
    s = math.sin(math.radians(d2_deg))
    th = phis / s if abs(s) > 1e-12 else phis
    return np.mod(th, 2 * PI) / (2 * PI)


def inverse_th4(P, d2_deg, rp):
    d2 = math.radians(d2_deg)
    rs = np.hypot(P[:, 0], P[:, 1])
    zs = P[:, 2]
    phis = np.arctan2(P[:, 1], P[:, 0])
    phis = np.where(phis < 0, phis + 2 * PI, phis)
    r = np.hypot(rs, zs)
    sd = math.sin(d2)
    th = np.mod(phis / sd, 2 * PI)
    Rs = (r / rp) ** (1.0 / sd)
    return Rs * np.cos(th), Rs * np.sin(th)


def w_average(P, rp, d2_deg, n_dirs=8):
    """模拟 AverageProbabilitySequence: 多方向概率场的平均"""
    acc = np.zeros(len(P))
    for k in range(n_dirs):
        v3 = v3_from_angles(d2_deg, k * 360.0 / n_dirs)
        p, _ = compute_probabilities(P, v3, rp, d2_deg)
        acc += p
    return acc / n_dirs


def kernel_loo_r2(X, w, sigma_frac=0.5, lam=1e-3):
    """核岭回归 + LOO 决定系数
       ŷ = H w,  H = K(K+λI)^{-1},  标准 LOO 残差 = (w-ŷ)/(1-diag H)
    """
    n = len(w)
    X = np.asarray(X, float)
    Xn = (X - X.mean(0)) / (X.std(0) + 1e-12)
    D2 = np.sum((Xn[:, None, :] - Xn[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(D2, np.nan)
    med = np.nanmedian(D2)                      # 只用非对角元素估计尺度
    if not np.isfinite(med) or med <= 0:
        return float('nan')
    sig = sigma_frac * math.sqrt(med)
    D2 = np.nan_to_num(D2, nan=0.0)
    K = np.exp(-D2 / (2.0 * sig ** 2))
    A = K + lam * np.eye(n)
    try:
        Ainv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return float('nan')
    H = K @ Ainv
    hii = np.clip(np.diag(H), 0.0, 1.0 - 1e-12)
    wc = w - w.mean()
    yhat = H @ wc
    loo = (wc - yhat) / (1.0 - hii)
    ss = float(np.sum(wc ** 2))
    if not (ss > 0):
        return float('nan')
    return 1.0 - float(np.sum(loo ** 2)) / ss


def main():
    print("=" * 104)
    print("w 的变量依赖性分析 (核岭回归 LOO_R2; 越接近 1 说明该特征集足以重建 w)")
    print("=" * 104)
    for fn in ["points.json", "points1.json", "pyjson.json"]:
        P = load_points(fn)
        rp = float(np.mean(np.linalg.norm(P, axis=1)))
        print(f"\n########## {fn}  n={len(P)} rp={rp:.4f} ##########")
        for d2d in [15.0, 45.0, 75.0]:
            theta = ring_theta(P, d2d)
            ur, ui = inverse_th4(P, d2d, rp)
            rho = np.hypot(ur, ui)
            r3 = np.linalg.norm(P, axis=1)
            w = w_average(P, rp, d2d, n_dirs=8)

            sets = {
                "θ (环坐标)":        theta.reshape(-1, 1),
                "ρ (=|uv|)":          rho.reshape(-1, 1),
                "r (=|P|)":           r3.reshape(-1, 1),
                "θ,ρ":               np.stack([theta, rho], 1),
                "θ,ρ,r":             np.stack([theta, rho, r3], 1),
                "(x,y,z) 位置":       P,
                "(x,y,z)+θ":         np.column_stack([P, theta]),
                "uv 实虚":            np.stack([ur, ui], 1),
                "θ,ρ,uv实虚":         np.stack([theta, rho, ur, ui], 1),
            }
            print(f"\n--- d2={d2d}°  (w: 均值 {w.mean():.4f} std {w.std():.4f}) ---")
            print(f"  {'特征集':>14s} {'维度':>4s} {'LOO_R2':>9s}")
            for name, X in sets.items():
                r2 = kernel_loo_r2(np.asarray(X, float), w)
                print(f"  {name:>14s} {X.shape[1]:4d} {r2:9.4f}")


if __name__ == "__main__":
    main()
