# -*- coding: utf-8 -*-
"""
验证: 用"uvs 上的局部二次拟合"替换 RBF(+3次数值梯度)

RBF 实际需求(从调用点提取): 把 N 个离散标量 w 变成 uvs 上可求一/二阶导的光滑场
  - 不需在新点预测 (Predict 只在原中心点调用 = 自插值)
  - 导数在 uvs 上求 (不是 S3)
  - 下游只需要 gradW 与 Hessian(H_uu,H_uv,H_vv)

对比:
  原: RBF(X∈S³, σ=0.8·avgAngle 或修正σ) -> w_pred -> NumericalGradient×3 -> Hessian
  新: uvs 上局部二次拟合 -> (w, gradW, Hessian) 一次得到, 解析

指标: 耗时 / Hessian 噪声稳定性(加 1% 噪声后相关性) / 量级
"""
import json
import math
import time

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
    return Rs * np.cos(th) + 1j * Rs * np.sin(th)


def build_X_from_uv(uv, theta2=0.0):
    """与 MapUVToS3 同构: (φ,Θ,ψ) -> S³,  Θ 取常数(因 V2 只贡献全局常数)"""
    theta1 = np.mod(np.angle(uv), 2 * PI)
    rho = np.abs(uv)
    meanR = rho.mean()
    ratio = np.clip(rho / (meanR + 1e-6), 0.1, 10.0)
    psi = PI / 2 * (ratio - 1) / (ratio + 1)
    phi, Th = theta1, theta2
    X = np.stack([np.cos(psi) * np.cos(Th) * np.cos(phi),
                  np.cos(psi) * np.cos(Th) * np.sin(phi),
                  np.cos(psi) * np.sin(Th),
                  np.sin(psi)], axis=1)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def numerical_gradient(uv, values, K=12):
    """与 C# NumericalGradient 等价: 每点 K 近邻拟合 (u,v,1) -> (dw/du, dw/dv)"""
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


def rbf_pipeline(uv, X, w, sigma, lam=1e-3):
    """原方法: S3 核 RBF 自插值 -> 3 次数值梯度 -> Hessian"""
    t0 = time.time()
    D = np.clip(X @ X.T, -1, 1); TH = np.arccos(D)
    K = np.exp(-(TH ** 2) / (2 * sigma ** 2))
    n = len(X)
    A = K + lam * np.eye(n)
    a = np.linalg.solve(A, w - w.mean())
    w_pred = w.mean() + K @ a
    t_rbf = time.time() - t0

    t0 = time.time()
    gW = numerical_gradient(uv, w_pred, 12)
    du, dv = gW[:, 0], gW[:, 1]
    gdu = numerical_gradient(uv, du, 12)
    gdv = numerical_gradient(uv, dv, 12)
    Huu, Huv, Hvv = gdu[:, 0], gdu[:, 1], gdv[:, 1]
    t_grad = time.time() - t0
    return dict(w=w_pred, gW=gW, H=(Huu, Huv, Hvv),
                t_rbf=t_rbf, t_grad=t_grad, t_total=t_rbf + t_grad)


def local_quad_fit(uv, w, k=16, ridge=1e-8):
    """新方法: uvs 上局部二次拟合, 一次得到 w / 一阶导 / Hessian"""
    t0 = time.time()
    u, v = uv.real, uv.imag
    n = len(uv); pts = np.stack([u, v], 1)
    wp = np.zeros(n); gW = np.zeros((n, 2)); H = np.zeros((n, 3))
    for i in range(n):
        d = np.linalg.norm(pts - pts[i], axis=1)
        order = np.argsort(d)
        nbr = [j for j in order if d[j] > 1e-12][:k]
        if len(nbr) < 6:
            wp[i] = w[i]; continue
        du = u[nbr] - u[i]; dv = v[nbr] - v[i]
        A = np.stack([np.ones(len(nbr)), du, dv,
                      0.5 * du ** 2, du * dv, 0.5 * dv ** 2], 1)
        AtA = A.T @ A + ridge * np.eye(6)
        try:
            coef = np.linalg.solve(AtA, A.T @ w[nbr])
        except np.linalg.LinAlgError:
            wp[i] = w[i]; continue
        wp[i] = coef[0]
        gW[i] = (coef[1], coef[2])
        H[i] = (coef[3], coef[4], coef[5])      # H_uu, H_uv, H_vv (解析)
    t = time.time() - t0
    return dict(w=wp, gW=gW, H=(H[:, 0], H[:, 1], H[:, 2]),
                t_rbf=0.0, t_grad=0.0, t_total=t)


def main():
    print("=" * 112)
    print("替换验证: RBF(+3次数值梯度)  vs  uvs 上局部二次拟合")
    print("=" * 112)
    rng = np.random.default_rng(0)
    for fn in ["points.json", "points1.json", "pyjson.json"]:
        P = load_points(fn)
        rp = float(np.mean(np.linalg.norm(P, axis=1)))
        print(f"\n########## {fn}  n={len(P)} ##########")
        for d2d in [15.0, 45.0]:
            uv = inverse_th4(P, d2d, rp)
            X = build_X_from_uv(uv)
            # w: 概率场多方向平均 (模拟 AverageProbabilitySequence)
            acc = np.zeros(len(P))
            for k in range(8):
                v3 = v3_from_angles(d2d, k * 45.0)
                p, _ = compute_probabilities(P, v3, rp, d2d)
                acc += p
            w = acc / 8.0

            # sigma: 原实现 0.8*avgAngle 与 修正近邻尺度
            D = np.clip(X @ X.T, -1, 1); TH = np.arccos(D)
            iu = np.triu_indices(len(X), 1)
            avg = TH[iu].mean()
            nn = np.array([np.min(np.delete(TH[i], i)) for i in range(len(X))])
            sig_orig = 0.8 * avg
            sig_fix = np.median(nn)

            res_orig = rbf_pipeline(uv, X, w, sig_orig)
            res_fix = rbf_pipeline(uv, X, w, sig_fix)
            res_loc = local_quad_fit(uv, w, k=16)

            print(f"\n--- d2={d2d}°  (σ_orig={sig_orig:.4f}, σ_fix={sig_fix:.5f}) ---")
            print(f"  {'方法':>16s} {'耗时(ms)':>10s} {'|H|中位':>10s} "
                  f"{'gradW 中位':>11s} {'|H| 噪声相关性':>14s} {'gradW 噪声相关性':>16s}")

            def noise_stability(fn_run, scale=0.01):
                """给 w 加 1% 噪声, 看重算结果与原结果的相关性"""
                h1 = np.concatenate(fn_run(w)["H"])
                g1 = fn_run(w)["gW"].ravel()
                wn = w + scale * w.std() * rng.standard_normal(len(w))
                h2 = np.concatenate(fn_run(wn)["H"])
                g2 = fn_run(wn)["gW"].ravel()
                ch = np.corrcoef(h1, h2)[0, 1]
                cg = np.corrcoef(g1, g2)[0, 1]
                return ch, cg

            for name, r in [("RBF(σ 原实现)", res_orig), ("RBF(σ 修正)", res_fix),
                            ("局部二次拟合", res_loc)]:
                Hm = np.median(np.abs(np.concatenate(r["H"])))
                gm = np.median(np.abs(r["gW"].ravel()))
                print(f"  {name:>16s} {r['t_total']*1000:10.1f} {Hm:10.4f} {gm:11.4f}",
                      end="")
                if name == "RBF(σ 原实现)":
                    ch, cg = noise_stability(lambda ww: rbf_pipeline(uv, X, ww, sig_orig))
                elif name == "RBF(σ 修正)":
                    ch, cg = noise_stability(lambda ww: rbf_pipeline(uv, X, ww, sig_fix))
                else:
                    ch, cg = noise_stability(lambda ww: local_quad_fit(uv, ww, k=16))
                print(f" {ch:14.4f} {cg:16.4f}")


if __name__ == "__main__":
    main()
