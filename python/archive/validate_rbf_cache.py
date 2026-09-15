# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
验证优化: RBF 的 K 矩阵只依赖 d2 (不依赖 v3/fj) → 分解可缓存
  a) 每样本重新分解 (原版)
  b) 同一 d2 下缓存 LU/Cholesky 分解, fj 变化只回代
对比: w_pred 是否完全一致 + 耗时
"""
import json
import math
import time

import numpy as np
import scipy.linalg as sla

from forward_sjy import inverse_th4, v3_from_angles, compute_probabilities
from validate_rbf_replacement import map_uv_to_s3, kernel_matrix


def rbf_solve_direct(X, delta, sigma, lam):
    K = kernel_matrix(X, sigma)
    N = len(X)
    A = np.zeros((N + 1, N + 1))
    A[:N, :N] = K + lam * np.eye(N)
    A[:N, N] = 1.0
    A[N, :N] = 1.0
    b = np.concatenate([delta, [0.0]])
    sol = np.linalg.solve(A, b)
    return sol[:N], sol[N], K


def make_operator(X, sigma, lam):
    """缓存: 返回 (lu_piv, K, N) 供后续回代复用"""
    K = kernel_matrix(X, sigma)
    N = len(X)
    A = np.zeros((N + 1, N + 1))
    A[:N, :N] = K + lam * np.eye(N)
    A[:N, N] = 1.0
    A[N, :N] = 1.0
    lu, piv = sla.lu_factor(A)
    return lu, piv, K


def rbf_predict_cached(lu, piv, K, delta):
    N = K.shape[0]
    b = np.concatenate([delta, [0.0]])
    sol = sla.lu_solve((lu, piv), b)
    w, c0 = sol[:N], sol[N]
    return c0 + K @ w


def main():
    with open(r"C:\Users\23128\My project (2)\Assets\Resources\points.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    P = np.array([np.array(p, float) for p in raw], float)
    rp = float(np.mean(np.linalg.norm(P, axis=1)))
    lam = 1e-3

    for d2_deg in [15.0, 30.0, 50.0]:
        uvs = np.array([inverse_th4(p, d2_deg, rp) for p in P])
        fjs = np.linspace(0, 350, 10)
        probs = []
        for fj in fjs:
            v3 = v3_from_angles(d2_deg, float(fj))
            prob, _ = compute_probabilities(P, v3, rp, d2_deg)
            probs.append(prob)
        X = map_uv_to_s3(uvs, probs[0])
        D = np.clip(X @ X.T, -1, 1)
        TH = np.arccos(D)
        iu = np.triu_indices(len(X), 1)
        sigma = max(TH[iu].mean() * 0.8, 1e-6)

        # a) 每样本重新分解 + 解
        t0 = time.time()
        out_a = []
        for pr in probs:
            w, c0, K = rbf_solve_direct(X, pr, sigma, lam)
            out_a.append(c0 + K @ w)
        t_a = (time.time() - t0) / len(probs)

        # b) 缓存分解
        t0 = time.time()
        lu, piv, K = make_operator(X, sigma, lam)
        t_factor = time.time() - t0
        t0 = time.time()
        out_b = [rbf_predict_cached(lu, piv, K, pr) for pr in probs]
        t_b = (time.time() - t0) / len(probs)

        diff = max(np.abs(a - b).max() for a, b in zip(out_a, out_b))
        rel = max(np.abs(a - b).max() / (np.abs(a).max() + 1e-30) for a, b in zip(out_a, out_b))
        print(f"d2={d2_deg:5.1f}°  a)逐次分解 {t_a*1000:8.2f}ms/样本 | "
              f"b)缓存复用 {t_b*1000:8.3f}ms/样本 (分解一次 {t_factor*1000:.2f}ms) | "
              f"加速 {t_a/max(t_b,1e-9):7.1f}x | 结果最大差 {diff:.2e} (相对 {rel:.2e})")

    print("\n结论: 若结果差 ≈ 0 且加速显著 → 可在采样/逆问题迭代中按 d2 缓存 RBF 分解")
    print("      K 只依赖 d2 (uvs 与 V2 都不含 v3), 只有 w 依赖 v3")


if __name__ == "__main__":
    main()
