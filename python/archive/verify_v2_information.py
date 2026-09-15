# -*- coding: utf-8 -*-
"""
验证: V2 (曲率半径 I / 透镜体积 V) 是否携带 w 的信息?
  若携带 -> 让 V2 逐点参与 X 才有意义 (例如把 theta2 从全局常数改成逐点)
  若不携带 -> V2 整段计算可以删掉

V2 来源 (LocalLensVolumeExtractor.ComputeAllFeatures 的语义):
  J[i] = pointsB[i] - pointsA[i]      雅可比场(两条路径之差)
  ds[i] = 两路径平均弧长步长
  ∇_vJ  = 中心差分
  R     = ||J|| / ||∇_vJ||            曲率半径
  I_arr = R
  x     = 1 - d²/(4R²)
  V_arr = R⁴ · I_x(5/2, 1/2)          不完全 Beta
  这里用 J[i] = P[i+1]-P[i] (切向差) 作为代理路径对
"""
import json
import math

import numpy as np
from scipy.special import betainc

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
    r = np.hypot(rs, zs); sd = math.sin(d2)
    th = np.mod(phis / sd, 2 * PI)
    Rs = (r / rp) ** (1.0 / sd)
    return Rs * np.cos(th) + 1j * Rs * np.sin(th)


def compute_all_features(P):
    """LocalLensVolumeExtractor.ComputeAllFeatures 的代理实现"""
    n = len(P)
    # 两条"路径": A = P, B = P 的循环后移一位 (切向差)
    A = P
    B = np.roll(P, -1, axis=0)
    J = B - A
    d = np.linalg.norm(J, axis=1)
    dsA = np.linalg.norm(np.diff(np.vstack([A, A[:1]]), axis=0), axis=1)
    dsB = np.linalg.norm(np.diff(np.vstack([B, B[:1]]), axis=0), axis=1)
    ds = 0.5 * (dsA + dsB)
    eps = 1e-12
    I_arr = np.zeros(n); V_arr = np.zeros(n)
    for i in range(n):
        if i == 0:
            step = max(ds[1], eps)
            gradJ = (J[1] - J[0]) / step
        elif i == n - 1:
            step = max(ds[n - 1], eps)
            gradJ = (J[n - 1] - J[n - 2]) / step
        else:
            step = max(0.5 * (ds[i] + ds[i + 1]), eps)
            gradJ = (J[i + 1] - J[i - 1]) / (2.0 * step)
        normGrad = np.linalg.norm(gradJ) + eps
        R = max(d[i] / normGrad, eps)
        I_arr[i] = R
        x = 1.0 - (d[i] ** 2) / (4.0 * R * R)
        x = min(max(x, 0.0), 1.0)
        V_arr[i] = R ** 4 * betainc(2.5, 0.5, x)      # 不完全 Beta
    return I_arr, V_arr


def kernel_loo_r2(X, w, sigma_frac=0.5, lam=1e-3):
    n = len(w)
    X = np.asarray(X, float)
    Xn = (X - X.mean(0)) / (X.std(0) + 1e-12)
    D2 = np.sum((Xn[:, None, :] - Xn[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(D2, np.nan)
    med = np.nanmedian(D2)
    if not np.isfinite(med) or med <= 0:
        return float('nan')
    sig = sigma_frac * math.sqrt(med)
    D2 = np.nan_to_num(D2, nan=0.0)
    K = np.exp(-D2 / (2 * sig ** 2))
    A = K + lam * np.eye(n)
    try:
        H = K @ np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return float('nan')
    hii = np.clip(np.diag(H), 0, 1 - 1e-12)
    wc = w - w.mean()
    loo = (wc - H @ wc) / (1 - hii)
    ss = float(np.sum(wc ** 2))
    return 1.0 - float(np.sum(loo ** 2)) / ss if ss > 0 else float('nan')


def main():
    print("=" * 100)
    print("V2 是否携带 w 的信息 (LOO_R2)")
    print("=" * 100)
    for fn in ["points.json", "points1.json", "pyjson.json"]:
        P = load_points(fn)
        rp = float(np.mean(np.linalg.norm(P, axis=1)))
        I_arr, V_arr = compute_all_features(P)
        print(f"\n########## {fn}  n={len(P)} ##########")
        print(f"  I: 范围 [{I_arr.min():.4g},{I_arr.max():.4g}]  "
              f"V: 范围 [{V_arr.min():.4g},{V_arr.max():.4g}]")
        for d2d in [15.0, 45.0, 75.0]:
            uv = inverse_th4(P, d2d, rp)
            rho = np.abs(uv)
            r3 = np.linalg.norm(P, axis=1)
            theta = np.mod(np.angle(uv), 2 * PI) / (2 * PI)
            In = I_arr / (I_arr.max() + 1e-30)
            Vn = V_arr / (V_arr.max() + 1e-30)
            theta2_pts = np.arctan2(Vn, In)          # 逐点版 theta2
            theta2_glb = np.arctan2(np.sin(theta2_pts).mean(),
                                    np.cos(theta2_pts).mean())  # 全局版(圆平均)
            # w: 多方向概率平均
            acc = np.zeros(len(P))
            for k in range(8):
                v3 = v3_from_angles(d2d, k * 45.0)
                p, _ = compute_probabilities(P, v3, rp, d2d)
                acc += p
            w = acc / 8.0

            print(f"\n  --- d2={d2d}° ---")
            sets = {
                "I 单独":              In.reshape(-1, 1),
                "V 单独":              Vn.reshape(-1, 1),
                "(I,V)":               np.stack([In, Vn], 1),
                "theta2 逐点":          theta2_pts.reshape(-1, 1),
                "r=|P| 单独":          r3.reshape(-1, 1),
                "(I,V,r)":             np.stack([In, Vn, r3], 1),
                "θ(环坐标) 单独":       theta.reshape(-1, 1),
                "theta2 全局(常数)":    np.full((len(P), 1), theta2_glb),
                "(r, theta2逐点)":     np.stack([r3, theta2_pts], 1),
            }
            print(f"    {'特征集':>18s} {'维度':>4s} {'LOO_R2':>9s}")
            for name, X in sets.items():
                r2 = kernel_loo_r2(X, w)
                mark = "  ←" if r2 > 0.3 else ""
                print(f"    {name:>18s} {X.shape[1]:4d} {r2:9.4f}{mark}")
        print()


if __name__ == "__main__":
    main()
