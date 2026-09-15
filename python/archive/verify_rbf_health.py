# -*- coding: utf-8 -*-
"""
检查 RBF 本身的计算是否健康

RBF 的关键参数:
  sigma = 0.8 * avgAngle        (avgAngle = 所有点对测地角的平均)
  lambda = 1e-3
  K[i,j] = exp(-theta_ij^2 / (2 sigma^2))
  (K+λI)a + c0 = w,   w_pred = c0 + K a

健康标准:
  1) sigma 应与"近邻尺度"同量级; 若 sigma >> 最近邻角 -> 核无局部性
  2) K 的特征值谱应有明显衰减结构(少数大特征值 + 长尾), 而不是"接近常数矩阵"
  3) K 的条件数不应爆炸到必须靠 λ 救命
  4) 自插值 w_pred 不应退化成"全局均值"或"原样复制"

X 的构造(与 MapUVToS3 同构; V2 只贡献两个全局常数, 这里用 mean|uv| 代理 meanR_V2):
  theta1 = arg(uv),  theta2 = 0 (常数),  theta3 = π/2 (ρ-1)/(ρ+1), ρ=|uv|/meanR
  X = (cosψcosΘcosφ, cosψcosΘsinφ, cosψsinΘ, sinψ)   Θ=0 -> (cosψcosφ, cosψsinφ, 0, sinψ)
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


def inverse_th4(P, d2_deg, rp):
    d2 = math.radians(d2_deg)
    rs = np.hypot(P[:, 0], P[:, 1]); zs = P[:, 2]
    phis = np.arctan2(P[:, 1], P[:, 0]); phis = np.where(phis < 0, phis + 2 * PI, phis)
    r = np.hypot(rs, zs)
    sd = math.sin(d2)
    th = np.mod(phis / sd, 2 * PI)
    Rs = (r / rp) ** (1.0 / sd)
    return Rs * np.cos(th), Rs * np.sin(th)


def build_X(P, d2_deg, rp, theta2=0.0):
    ur, ui = inverse_th4(P, d2_deg, rp)
    theta1 = np.mod(np.arctan2(ui, ur), 2 * PI)
    rho = np.hypot(ur, ui)
    meanR = rho.mean()
    ratio = np.clip(rho / (meanR + 1e-6), 0.1, 10.0)
    theta3 = PI / 2 * (ratio - 1) / (ratio + 1)
    phi, Th, psi = theta1, theta2, theta3
    X = np.stack([np.cos(psi) * np.cos(Th) * np.cos(phi),
                  np.cos(psi) * np.cos(Th) * np.sin(phi),
                  np.cos(psi) * np.sin(Th),
                  np.sin(psi)], axis=1)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def geo_angles(X):
    D = np.clip(X @ X.T, -1, 1)
    return np.arccos(D)


def analyze(X, w, tag, sigma_mode="orig", lam=1e-3):
    TH = geo_angles(X)
    n = len(X)
    iu = np.triu_indices(n, 1)
    avg = TH[iu].mean()
    nn = np.array([np.min(np.delete(TH[i], i)) for i in range(n)])   # 最近邻角
    if sigma_mode == "orig":
        sigma = 0.8 * avg                          # 原实现
    elif sigma_mode == "nn":
        sigma = np.median(nn)                      # 近邻尺度
    else:
        sigma = sigma_mode
    K = np.exp(-(TH ** 2) / (2 * sigma ** 2))
    ev = np.linalg.eigvalsh(K)[::-1]
    ev_pos = np.maximum(ev, 0)
    tot = ev_pos.sum()
    acc = np.cumsum(ev_pos) / (tot + 1e-300)
    r90 = int(np.searchsorted(acc, 0.90) + 1)
    r99 = int(np.searchsorted(acc, 0.99) + 1)
    condK = ev_pos[0] / (ev_pos[-1] + 1e-300)
    # 自插值
    A = K + lam * np.eye(n)
    a = np.linalg.solve(A, w - w.mean())
    w_pred = w.mean() + K @ a
    rms = math.sqrt(np.mean((w_pred - w) ** 2))
    std = w.std()
    shrink = np.std(w_pred) / (std + 1e-300)
    return dict(avg=avg, nn_med=np.median(nn), sigma=sigma,
                r90=r90, r99=r99, condK=condK, shrink=shrink, rms=rms, std=std)


def main():
    print("=" * 108)
    print("RBF 计算健康度检查 (X 用 InverseTh4 的 (φ,ψ) 构造, σ 分别取 原实现 / 近邻尺度)")
    print("=" * 108)
    for fn in ["points.json", "points1.json", "pyjson.json"]:
        P = load_points(fn)
        rp = float(np.mean(np.linalg.norm(P, axis=1)))
        print(f"\n########## {fn}  n={len(P)} rp={rp:.4f} ##########")
        for d2d in [15.0, 45.0, 75.0]:
            X = build_X(P, d2d, rp)
            v3 = v3_from_angles(d2d, 0.0)
            w, _ = compute_probabilities(P, v3, rp, d2d)
            w = w.astype(float)
            print(f"\n--- d2={d2d}° ---")
            print(f"  {'模式':>10s} {'σ':>9s} {'平均角':>8s} {'近邻角中位':>10s} "
                  f"{'σ/近邻':>8s} {'90%秩':>6s} {'99%秩':>6s} {'cond(K)':>11s} "
                  f"{'收缩率':>7s} {'RMS(wp-w)/std':>13s}")
            for mode, name in [("orig", "原实现"), ("nn", "近邻尺度")]:
                r = analyze(X, w, name, mode)
                print(f"  {name:>10s} {r['sigma']:9.5f} {r['avg']:8.5f} {r['nn_med']:10.5f} "
                      f"{r['sigma']/r['nn_med']:8.2f} {r['r90']:6d} {r['r99']:6d} "
                      f"{r['condK']:11.3e} {r['shrink']:7.4f} {r['rms']/r['std']:13.4f}")


if __name__ == "__main__":
    main()
