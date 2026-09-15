# -*- coding: utf-8 -*-
"""
验证路径 B: 把球面 RBF 换成"周期(傅里叶)插值"是否可行

思路: 改善后的环坐标 theta ∈ [0,1) 是周期坐标 => w(theta) 是周期函数
  RBF : O(N^3) 解 (K+λI)a=w, 预测 O(N), Hessian 需两次数值梯度(有噪声)
  傅里叶: w(θ) = Σ c_k e^{i2πkθ}, 最小二乘解 c, 二阶导 = Σ -4π²k² c_k e^{i2πkθ} (解析)

指标:
  1) 频谱 |c_k| 衰减速度  -> 少数谐波能否重建 w
  2) 截断傅里叶重建误差 vs K
  3) 对照: 高斯核自插值 (RBF 在 θ 上的类比) 的重建误差
  4) 二阶导: 傅里叶解析 vs 核插值/数值差分的噪声水平
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
    """改善后的环坐标: thetas = phis / sin(d2)  (与 InverseTh4 展开一致)"""
    phis = np.arctan2(P[:, 1], P[:, 0])
    phis = np.where(phis < 0, phis + 2 * PI, phis)
    s = math.sin(math.radians(d2_deg))
    th = phis / s if abs(s) > 1e-12 else phis
    th = np.mod(th, 2 * PI)
    return th / (2 * PI)          # 归一化到 [0,1)


def fourier_fit(theta, w, K):
    """最小二乘周期拟合: w(θ) ≈ Σ_{k=-K}^{K} c_k e^{i2πkθ}"""
    ks = np.arange(-K, K + 1)
    A = np.exp(2j * PI * np.outer(theta, ks))
    c, *_ = np.linalg.lstsq(A, w.astype(complex), rcond=None)
    return c, ks, A


def gaussian_self_interp(theta, w, sigma, lam=1e-3):
    """RBF(自插值)在 θ 上的类比: 核 K(Δθ)=exp(-Δθ²/2σ²), 解 (K+λI)a=w, 预测 at 中心"""
    d = np.abs(theta[:, None] - theta[None, :])
    d = np.minimum(d, 1.0 - d)                 # 周期距离 ∈ [0,0.5]
    K = np.exp(-(d ** 2) / (2 * sigma ** 2))
    n = len(theta)
    A = K + lam * np.eye(n)
    a = np.linalg.solve(A, w)
    return K @ a, K


def second_deriv_numeric(theta, vals, h=1e-3):
    """不均匀网格上的二阶导 (中心差分 + 近邻插值)"""
    idx = np.argsort(theta)
    ts, vs = theta[idx], vals[idx]
    # 周期最近邻的均匀插值
    n = len(ts)
    d2 = np.zeros(n)
    for i in range(n):
        ip = ts[(i + 1) % n] if i + 1 < n else ts[0] + 1.0
        im = ts[i - 1] if i > 0 else ts[-1] - 1.0
        d2[i] = (vs[(i + 1) % n] - 2 * vs[i] + vs[i - 1]) / ((0.5 * (ip - im)) ** 2)
    out = np.zeros(n)
    out[idx] = d2
    return out


def main():
    print("=" * 100)
    print("路径 B 验证: 周期(傅里叶)插值 vs 核(RBF)自插值")
    print("=" * 100)
    for fn in ["points.json", "points1.json", "pyjson.json"]:
        P = load_points(fn)
        rp = float(np.mean(np.linalg.norm(P, axis=1)))
        print(f"\n########## {fn}  n={len(P)} rp={rp:.4f} ##########")
        for d2d in [15.0, 45.0, 75.0]:
            # 环坐标 (改善后) 与标量场 w (用概率场作为 RBF 目标 w 的代理)
            theta = ring_theta(P, d2d)
            v3 = v3_from_angles(d2d, 0.0)
            w, _ = compute_probabilities(P, v3, rp, d2d)
            w = w.astype(float)

            print(f"\n--- d2={d2d}° ---")
            print(f"  θ: 范围 [{theta.min():.4f},{theta.max():.4f}]  "
                  f"w: 均值 {w.mean():.4f} 标准差 {w.std():.4f} 范围 [{w.min():.4f},{w.max():.4f}]")

            # ---- 1) 频谱衰减 ----
            Kmax = 24
            c, ks, A = fourier_fit(theta, w, Kmax)
            mag = np.abs(c)
            # 按 |k| 聚合
            byk = {}
            for kk, mm in zip(ks, mag):
                byk[abs(kk)] = byk.get(abs(kk), 0.0) + mm
            print(f"  频谱(按|k|聚合, 前 8 个): " +
                  " ".join(f"k={k}:{byk.get(k,0):.3e}" for k in range(0, 8)))
            # 能量占比
            e0 = byk.get(0, 0.0) ** 2
            tot = sum(v * v for v in byk.values()) + 1e-300
            cum = {}
            acc = 0.0
            for k in sorted(byk):
                acc += byk[k] ** 2
                cum[k] = acc / tot
            k90 = next((k for k in sorted(cum) if cum[k] >= 0.90), None)
            k99 = next((k for k in sorted(cum) if cum[k] >= 0.99), None)
            print(f"  累积能量: 90% 需要 |k|≤{k90}, 99% 需要 |k|≤{k99}")

            # ---- 2) 截断重建误差 (留一交叉验证更能说明泛化) ----
            print(f"  {'K':>3s} {'傅里叶重建误差':>14s} {'傅里叶LOO误差':>14s} {'核自插值误差':>13s}")
            n = len(w)
            for K in [1, 2, 3, 5, 8, 12]:
                cc, kks, AA = fourier_fit(theta, w, K)
                rec = (AA @ cc).real
                err = np.linalg.norm(w - rec) / np.linalg.norm(w)
                # LOO: 逐个剔除求解 (慢但 N=200 可接受)
                loo_err = 0.0
                for i in range(n):
                    mask = np.ones(n, bool)
                    mask[i] = False
                    ci, *_ = np.linalg.lstsq(AA[mask], w[mask].astype(complex), rcond=None)
                    pred_i = (AA[i] @ ci).real
                    loo_err += (pred_i - w[i]) ** 2
                loo_err = math.sqrt(loo_err / n) / (w.std() + 1e-12)
                # 核自插值 (σ 取 θ 上平均间距的量级)
                dsig = 0.8 * np.abs(np.subtract.outer(theta, theta)).min(axis=1).mean()
                _, KK = gaussian_self_interp(theta, w, max(dsig, 1e-3))
                kern_rec, _ = gaussian_self_interp(theta, w, max(dsig, 1e-3))
                kerr = np.linalg.norm(w - kern_rec) / np.linalg.norm(w)
                print(f"  {K:3d} {err:14.4e} {loo_err:14.4e} {kerr:13.4e}")

            # ---- 3) 二阶导: 傅里叶解析 vs 数值差分 ----
            cc, kks, AA = fourier_fit(theta, w, 8)
            d2_fourier = (AA @ (-4 * PI ** 2 * kks ** 2 * cc)).real
            d2_num = second_deriv_numeric(theta, w)
            ratio = np.std(d2_num) / (np.std(d2_fourier) + 1e-30)
            corr = np.corrcoef(d2_fourier, d2_num)[0, 1]
            print(f"  二阶导: 傅里叶解析 std={np.std(d2_fourier):.3e}  "
                  f"数值差分 std={np.std(d2_num):.3e}  比值={ratio:.3f}  相关={corr:.4f}")


if __name__ == "__main__":
    main()
