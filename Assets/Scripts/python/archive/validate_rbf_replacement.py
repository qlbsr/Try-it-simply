# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
验证: SphericalRBF 自插值能否被廉价平滑替代?
  w_pred 四种取法 → 同样下游 (Hessian → stiffnessProj → totalGrad → ABCD → tau)
  a) RBF 自插值 (原版, O(N^3))
  b) w 不平滑 (O(1))
  c) 邻域高斯平滑 k近邻 (O(N k))
  d) 全核平滑 (O(N^2), 不解方程)
指标: ABCD 相对误差 (以 a 为基准), tau 差, 耗时
"""
import json
import math
import time

import numpy as np

from forward_sjy import (inverse_th4, v3_from_angles, compute_probabilities,
                         numerical_gradient, fit_polynomial, tau_from_abcd, j_invariant)


def map_uv_to_s3(uvs, w):
    """简化版 MapUVToS3: theta1=arg(uv), theta2=圆平均arg(V2), theta3=径向比
    这里 V2 用 (|uv|, w) 归一化替代 (原版为透镜特征 I,V)"""
    uv = np.asarray(uvs, complex)
    theta1 = np.angle(uv) % (2 * np.pi)
    V2 = np.stack([np.abs(uv), np.asarray(w, float)], axis=1)
    V2 = V2 / (V2.max(0, keepdims=True) + 1e-30)
    phi = np.arctan2(V2[:, 1], V2[:, 0])
    theta2 = math.atan2(np.sin(phi).sum(), np.cos(phi).sum()) % (2 * np.pi)
    meanR = np.abs(V2).mean()
    rho = np.clip(np.abs(uv) / (meanR + 1e-12), 0.1, 10.0)
    theta3 = np.clip(np.pi / 2 * (rho - 1) / (rho + 1), -np.pi / 2, np.pi / 2)
    psi, th, ph = theta3, theta2, theta1
    X = np.stack([np.cos(psi) * np.cos(th) * np.cos(ph),
                  np.cos(psi) * np.cos(th) * np.sin(ph),
                  np.cos(psi) * np.sin(th),
                  np.sin(psi)], axis=1)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def kernel_matrix(X, sigma):
    D = np.clip(X @ X.T, -1, 1)
    TH = np.arccos(D)
    return np.exp(-(TH ** 2) / (2 * sigma ** 2))


def rbf_self_interp(X, delta, sigma, lam):
    """(K+λI)w + c0 = delta 带常数项 → 在中心上预测 (原版做法)"""
    K = kernel_matrix(X, sigma)
    N = len(X)
    A = np.zeros((N + 1, N + 1))
    A[:N, :N] = K + lam * np.eye(N)
    A[:N, N] = 1.0
    A[N, :N] = 1.0
    b = np.concatenate([delta, [0.0]])
    sol = np.linalg.solve(A, b)
    w, c0 = sol[:N], sol[N]
    return c0 + K @ w


def neighborhood_smooth(X, w, sigma, k=32):
    K = kernel_matrix(X, sigma)
    kk = min(k, K.shape[1])
    idx = np.argsort(-K, axis=1)[:, :kk]
    W = np.take_along_axis(K, idx, axis=1)
    W = W / (W.sum(1, keepdims=True) + 1e-30)
    return (W * w[idx]).sum(1)


def full_smooth(X, w, sigma):
    K = kernel_matrix(X, sigma)
    return K @ w / (K.sum(1) + 1e-30)


def downstream(P, d2_deg, fj_deg, rp, w_pred_mode, k=32):
    v3 = v3_from_angles(d2_deg, fj_deg)
    prob, info = compute_probabilities(P, v3, rp, d2_deg)
    uvs = np.array([inverse_th4(p, d2_deg, rp) for p in P])
    X = map_uv_to_s3(uvs, prob)
    # 平均测地角 → sigma
    D = np.clip(X @ X.T, -1, 1)
    TH = np.arccos(D)
    iu = np.triu_indices(len(X), 1)
    avg_ang = TH[iu].mean()
    sigma = max(avg_ang * 0.8, 1e-6)
    lam = 1e-3

    t0 = time.time()
    if w_pred_mode == "rbf":
        w_pred = rbf_self_interp(X, prob, sigma, lam)
    elif w_pred_mode == "none":
        w_pred = prob
    elif w_pred_mode == "nbr":
        w_pred = neighborhood_smooth(X, prob, sigma, k)
    elif w_pred_mode == "full":
        w_pred = full_smooth(X, prob, sigma)
    t_smooth = time.time() - t0

    # stiffnessProj = 0.5 * H * z, H 来自 w_pred 的 Hessian
    gradW = numerical_gradient(uvs, w_pred, 12)
    du = gradW[:, 0].copy()
    dv = gradW[:, 1].copy()
    g_du = numerical_gradient(uvs, du, 12)
    g_dv = numerical_gradient(uvs, dv, 12)
    H_uu, H_uv, H_vv = g_du[:, 0], g_du[:, 1], g_dv[:, 1]
    u, v = uvs.real, uvs.imag
    stiff = 0.5 * ((H_uu * u + H_uv * v) + 1j * (H_uv * u + H_vv * v))
    # AB_vals
    nrm = np.abs(uvs)
    nrm[nrm < 1e-12] = 1e-12
    ext = (uvs / nrm).real * v3[0] + (uvs / nrm).imag * v3[2]
    gE = numerical_gradient(uvs, ext, 12)
    AB = 0.5 * (gE[:, 0] - 1j * gE[:, 1])
    total = 6.0 * (stiff + AB)
    total = np.where(np.abs(total) > 70.0, 0.0 + 0.0j, total)
    A, B, C, D = fit_polynomial(uvs, total)
    tau, ratio, disc = tau_from_abcd(A, B, C, D)
    j = j_invariant(tau)
    return dict(ABCD=np.array([A, B, C, D]), tau=tau, j=j,
                sigma=sigma, t_smooth=t_smooth, total=total)


def main():
    with open(r"C:\Users\23128\My project (2)\Assets\Resources\points.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    P = np.array([np.array(p, float) for p in raw], float)
    rp = float(np.mean(np.linalg.norm(P, axis=1)))

    rng = np.random.default_rng(0)
    cases = [(float(rng.uniform(8, 60)), float(rng.uniform(0, 360))) for _ in range(15)]

    res = {m: [] for m in ["rbf", "none", "nbr", "full"]}
    times = {m: [] for m in res}
    for d2, fj in cases:
        for mode in res:
            r = downstream(P, d2, fj, rp, mode)
            res[mode].append(r)
            times[mode].append(r["t_smooth"])

    base = res["rbf"]
    print(f"{'模式':>6s} {'ABCD相对误差':>13s} {'|Δtau|':>10s} {'|j|相对差':>11s} {'平滑耗时':>9s}")
    for mode in ["rbf", "none", "nbr", "full"]:
        rel_ab, dtau, rel_j = [], [], []
        for a, b in zip(base, res[mode]):
            nb = np.linalg.norm(b["ABCD"])
            na = np.linalg.norm(a["ABCD"])
            rel_ab.append(np.linalg.norm(b["ABCD"] - a["ABCD"]) / (na + 1e-300))
            dtau.append(abs(b["tau"] - a["tau"]))
            rel_j.append(abs(b["j"] - a["j"]) / (abs(a["j"]) + 1e-300))
        print(f"{mode:>6s} {np.mean(rel_ab)*100:12.3f}% {np.mean(dtau):10.4f} "
              f"{np.mean(rel_j)*100:10.3f}% {np.mean(times[mode])*1000:8.2f}ms")
    print("\n基准 = rbf (原版自插值)。ABCD 相对误差 <1% 即认为可替代。")
    print(f"注: sigma 平均 {np.mean([r['sigma'] for r in base]):.4f} rad, N={len(P)}")


if __name__ == "__main__":
    main()
