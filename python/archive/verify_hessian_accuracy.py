# -*- coding: utf-8 -*-
"""
判决测试: 用已知解析函数比较两种方法的 Hessian 精度

在 uvs 上构造 w = f(ρ), ρ=|uv|   (径向函数, 与真实 w 的主导结构一致)
取 f(ρ) = exp(-ρ²/2), 则
  f'  = -ρ·exp(-ρ²/2)
  f'' = (ρ²-1)·exp(-ρ²/2)
解析 Hessian:
  H_uu = f''·u²/ρ² + f'·(1/ρ - u²/ρ³)
  H_uv = f''·u·v/ρ² - f'·u·v/ρ³
  H_vv = f''·v²/ρ² + f'·(1/ρ - v²/ρ³)
比较 RBF(+3次数值梯度) 与 局部二次拟合 的 Hessian 相对误差
"""
import json
import math

import numpy as np

from forward_sjy import compute_probabilities

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


def build_X_from_uv(uv, theta2=0.0):
    theta1 = np.mod(np.angle(uv), 2 * PI)
    rho = np.abs(uv); meanR = rho.mean()
    ratio = np.clip(rho / (meanR + 1e-6), 0.1, 10.0)
    psi = PI / 2 * (ratio - 1) / (ratio + 1)
    phi, Th = theta1, theta2
    X = np.stack([np.cos(psi)*np.cos(Th)*np.cos(phi), np.cos(psi)*np.cos(Th)*np.sin(phi),
                  np.cos(psi)*np.sin(Th), np.sin(psi)], 1)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def numerical_gradient(uv, values, K=12):
    uvx, uvy = uv.real, uv.imag
    vals = np.asarray(values, float)
    n = len(uv); grad = np.zeros((n, 2)); pts = np.stack([uvx, uvy], 1)
    for i in range(n):
        d = np.linalg.norm(pts - pts[i], axis=1)
        order = np.argsort(d)
        nbr = [j for j in order if d[j] > 1e-8][:K]
        if len(nbr) < 3: continue
        A = np.stack([uvx[nbr], uvy[nbr], np.ones(len(nbr))], 1)
        sol, *_ = np.linalg.lstsq(A, vals[nbr], rcond=None)
        grad[i] = (sol[0], sol[1])
    return grad


def rbf_hessian(uv, X, w, sigma, lam=1e-3):
    D = np.clip(X @ X.T, -1, 1); TH = np.arccos(D)
    K = np.exp(-(TH ** 2) / (2 * sigma ** 2))
    n = len(X)
    a = np.linalg.solve(K + lam * np.eye(n), w - w.mean())
    w_pred = w.mean() + K @ a
    gW = numerical_gradient(uv, w_pred, 12)
    gdu = numerical_gradient(uv, gW[:, 0], 12)
    gdv = numerical_gradient(uv, gW[:, 1], 12)
    return w_pred, np.stack([gdu[:, 0], gdu[:, 1], gdv[:, 1]], 1)


def local_hessian(uv, w, k=16, ridge=1e-8):
    u, v = uv.real, uv.imag
    n = len(uv); pts = np.stack([u, v], 1)
    wp = np.zeros(n); H = np.zeros((n, 3))
    for i in range(n):
        d = np.linalg.norm(pts - pts[i], axis=1)
        order = np.argsort(d); nbr = [j for j in order if d[j] > 1e-12][:k]
        if len(nbr) < 6:
            wp[i] = w[i]; continue
        du = u[nbr] - u[i]; dv = v[nbr] - v[i]
        A = np.stack([np.ones(len(nbr)), du, dv, 0.5*du**2, du*dv, 0.5*dv**2], 1)
        coef = np.linalg.solve(A.T @ A + ridge*np.eye(6), A.T @ w[nbr])
        wp[i] = coef[0]; H[i] = (coef[3], coef[4], coef[5])
    return wp, H


def main():
    print("=" * 104)
    print("判决测试: Hessian 精度 (w = exp(-ρ²/2), 解析 Hessian 为真值)")
    print("=" * 104)
    for fn in ["points.json", "pyjson.json"]:
        P = load_points(fn)
        rp = float(np.mean(np.linalg.norm(P, axis=1)))
        print(f"\n########## {fn}  n={len(P)} ##########")
        for d2d in [15.0, 45.0]:
            uv = inverse_th4(P, d2d, rp)
            u, v = uv.real, uv.imag
            rho = np.hypot(u, v)
            rho_s = rho / (rho.mean() + 1e-12) * 0.5      # 缩放到合适范围
            w = np.exp(-rho_s ** 2 / 2)
            fp = -rho_s * np.exp(-rho_s ** 2 / 2)
            fpp = (rho_s ** 2 - 1) * np.exp(-rho_s ** 2 / 2)
            # 注意: 对 rho_s 的导数要乘链式因子 s=0.5/mean(rho)
            s = 0.5 / (rho.mean() + 1e-12)
            fp_o = fp * s
            fpp_o = fpp * s * s
            rs = np.maximum(rho, 1e-12)
            Huu = fpp_o * u**2/rs**2 + fp_o * (1/rs - u**2/rs**3)
            Huv = fpp_o * u*v/rs**2 - fp_o * u*v/rs**3
            Hvv = fpp_o * v**2/rs**2 + fp_o * (1/rs - v**2/rs**3)
            H_exact = np.stack([Huu, Huv, Hvv], 1)

            X = build_X_from_uv(uv)
            D = np.clip(X @ X.T, -1, 1); TH = np.arccos(D)
            iu = np.triu_indices(len(X), 1)
            sig_fix = np.median([np.min(np.delete(TH[i], i)) for i in range(len(X))])

            _, H_rbf = rbf_hessian(uv, X, w, sig_fix)
            _, H_loc = local_hessian(uv, w, k=16)

            def relerr(H):
                num = np.linalg.norm(H - H_exact)
                den = np.linalg.norm(H_exact) + 1e-300
                # 逐分量相关
                corr = np.corrcoef(H.ravel(), H_exact.ravel())[0, 1]
                return num/den, corr

            e_rbf, c_rbf = relerr(H_rbf)
            e_loc, c_loc = relerr(H_loc)
            print(f"\n--- d2={d2d}° (σ_fix={sig_fix:.5f}) ---")
            print(f"  {'方法':>16s} {'H 相对误差':>12s} {'H 相关性':>10s} {'|H|中位(真)':>12s} "
                  f"{'|H|中位(算)':>12s}")
            print(f"  {'解析真值':>16s} {'-':>12s} {'1.0000':>10s} "
                  f"{np.median(np.abs(H_exact)):12.6f} {'-':>12s}")
            print(f"  {'RBF+数值梯度':>16s} {e_rbf:12.4f} {c_rbf:10.4f} {'':>12s} "
                  f"{np.median(np.abs(H_rbf)):12.6f}")
            print(f"  {'局部二次拟合':>16s} {e_loc:12.4f} {c_loc:10.4f} {'':>12s} "
                  f"{np.median(np.abs(H_loc)):12.6f}")


if __name__ == "__main__":
    main()
