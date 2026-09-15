# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
分析用户改动: MapDoubleConeToLeaf 里新增 v3 = (v1[0]-v1[1]).normalized
  1) 解析 v3_new 的 (d2, fj)
  2) 与旧 v3 (来自 (d2, PCA fj)) 对比下游 prob / ABCD / j
"""
import json
import math

import numpy as np

from forward_sjy import (inverse_th4, v3_from_angles, compute_probabilities,
                         numerical_gradient, fit_polynomial, tau_from_abcd, j_invariant)


def th2(theta, H):
    """C# sjy.th2 忠实"""
    sinRot = math.sin(theta)
    cosRot = math.cos(theta)
    tanX = -H / sinRot
    x = math.atan(tanX)
    c2 = np.array([x, H, tanX * cosRot])
    tanX_low = H / sinRot
    x_low = math.atan(tanX_low)
    c1 = np.array([x_low, H, tanX_low * cosRot])
    return c1, c2


def v3_from_leaf(rp, d2_deg, use_theta2=None):
    """按用户改动计算 v3 = normalize(v1[0]-v1[1])"""
    alpha = math.radians(d2_deg)
    h = rp * math.cos(alpha)
    R = rp * math.sin(alpha)
    H = math.pi * R
    theta = math.atan2(0.5 * h, H)
    th = theta if use_theta2 is None else use_theta2
    c1, c2 = th2(th, H)
    d = c1 - c2
    v = d / (np.linalg.norm(d) + 1e-30)
    d2_new = math.degrees(math.acos(np.clip(v[1], -1, 1)))
    fj_new = math.degrees(math.atan2(v[2], v[0])) % 360
    return v, d2_new, fj_new, dict(h=h, R=R, H=H, theta=theta)


def downstream(P, rp, v3, d2_deg):
    prob, info = compute_probabilities(P, v3, rp, d2_deg)
    uvs = np.array([inverse_th4(p, d2_deg, rp) for p in P])
    g = numerical_gradient(uvs, prob, 12)
    total = 6.0 * (0.5 * (g[:, 0] - 1j * g[:, 1]))
    total = np.where(np.abs(total) > 70.0, 0.0 + 0.0j, total)
    A, B, C, D = fit_polynomial(uvs, total)
    tau, ratio, disc = tau_from_abcd(A, B, C, D)
    j = j_invariant(tau)
    return prob, np.array([A, B, C, D]), tau, j


def main():
    with open(r"C:\Users\23128\My project (2)\Assets\Resources\points.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    P = np.array([np.array(p, float) for p in raw], float)
    rp = float(np.mean(np.linalg.norm(P, axis=1)))

    print("=== 1. 用户改动的 v3_new 解析值 ===")
    print(f"{'d2(输入)':>9s} {'h':>7s} {'H':>7s} {'theta(°)':>9s} {'v3_new':>26s} "
          f"{'d2_new(°)':>10s} {'fj_new(°)':>10s}")
    rows = {}
    for d2d in [10, 20, 30, 45, 60, 75]:
        v, d2n, fjn, geo = v3_from_leaf(rp, d2d)
        rows[d2d] = (v, d2n, fjn, geo)
        print(f"{d2d:9d} {geo['h']:7.3f} {geo['H']:7.3f} {math.degrees(geo['theta']):9.2f} "
              f"({v[0]:+.4f},{v[1]:+.4f},{v[2]:+.4f}) {d2n:10.1f} {fjn:10.2f}")

    print("\n→ d2_new 恒为 90.0°: 改动的 v3 永远躺在 x-z 平面 (赤道)")
    print("→ 与输入 d2 无关的锥角, 但方位角 fj_new 随 d2 变化")

    print("\n=== 2. 下游对比 (同一 d2 传入参数) ===")
    print(f"{'d2':>5s} {'旧v3(d2,PCA fj)':>22s} {'新v3(母线差)':>22s} "
          f"{'prob相关差':>10s} {'ABCD相对差':>11s} {'|Δtau|':>8s} {'j相对差':>9s}")
    for d2d in [15, 30, 50]:
        v_old = v3_from_angles(d2d, 0.0)           # 旧: (d2, fj=0) 代表
        v_new, d2n, fjn, _ = rows.get(d2d) or v3_from_leaf(rp, d2d)
        v_new = v3_from_leaf(rp, d2d)[0]
        p_old, ab_old, t_old, j_old = downstream(P, rp, v_old, d2d)
        p_new, ab_new, t_new, j_new = downstream(P, rp, v_new, d2d)
        pcorr = float(np.corrcoef(p_old, p_new)[0, 1])
        rel_ab = np.linalg.norm(ab_new - ab_old) / (np.linalg.norm(ab_old) + 1e-300)
        dtau = abs(t_new - t_old)
        rel_j = abs(j_new - j_old) / (abs(j_old) + 1e-300)
        print(f"{d2d:5d} ({v_old[0]:+.3f},{v_old[1]:+.3f},{v_old[2]:+.3f}) "
              f"({v_new[0]:+.3f},{v_new[1]:+.3f},{v_new[2]:+.3f}) "
              f"{pcorr:10.4f} {rel_ab*100:10.2f}% {dtau:8.4f} {rel_j*100:8.2f}%")

    print("\n=== 3. v3_new 自身随 d2 的变化 ===")
    print("d2 输入 → v3_new (d2=90 固定, 只 fj 变):")
    for d2d in [10, 20, 30, 45, 60, 75]:
        v, d2n, fjn, _ = rows[d2d]
        print(f"   d2={d2d:3d}° → v3_new=({v[0]:+.4f},{v[1]:+.4f},{v[2]:+.4f})  fj_new={fjn:6.2f}°")


if __name__ == "__main__":
    main()
