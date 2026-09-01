# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
PCA 原理 × 三角形: 高场 Σh_i² 就是点到轴距离平方和 (= 方差)
  每个点 p_i 与方向 v 的三角形高 h_i = |p_i × v| = |p_i|·sin∠(p_i,v)
  Σ h_i² = Σ |p_i×v|² = 点云到 v 轴的平方距离和
  PCA 主轴 v 使 Σ h_i² 最小 (投影方差最大 = 距离和最小)
  而 h_i 可从三角形边 a,b,2c 计算: h_i = (2·面积)/底 = 面积/c
  面积可由三边 (a,b,2c) 海伦公式! a,b 从概率反演可得!
验证: ① Σh_i²(v) 是否在 v=v_pca 时最小 (PCA 原理)
      ② 三角形高 vs 协方差主轴的一致性
      ③ 用高场能否无 PCA 定位主轴
"""
import json
import math

import numpy as np

import data_driven_axis as dd
import n2sjy2 as n2
import nsjy_algorithms as m

PYJSON = r"C:\Users\23128\My project (2)\Assets\Resources\pyjson.json"


def angle(u, v):
    return math.degrees(math.acos(np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1)))


def batch_probability(points, F1, F2, a):
    P = np.asarray(points, float)
    d1 = np.linalg.norm(P - np.asarray(F1, float), axis=1)
    d2 = np.linalg.norm(P - np.asarray(F2, float), axis=1)
    delta = d1 + d2 - 2.0 * a
    return np.exp(-np.abs(delta) / (2.0 * a))


def build_inv_table(p_i, c, a, n=7201):
    r0 = np.linalg.norm(p_i)
    th = np.linspace(0, 180, n)
    s = np.zeros(n)
    for i, t in enumerate(th):
        rad = math.radians(t)
        d1 = math.sqrt(r0 ** 2 + c ** 2 - 2 * r0 * c * math.cos(rad))
        d2 = math.sqrt(r0 ** 2 + c ** 2 + 2 * r0 * c * math.cos(rad))
        s[i] = math.exp(-abs(d1 + d2 - 2 * a) / (2 * a))
    return th, s


def main():
    with open(PYJSON, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    pts = [np.array([p["x"], p["y"], p["z"]], float) for p in raw]
    P = np.array(pts, float)
    rp = n2.compute_rp(pts)
    d2 = math.asin(math.cos(math.radians(30)) / math.pi)
    a = rp * (1 + math.sin(d2))
    e3 = math.cos(d2) * 2 / (1 + math.sin(d2))
    c = rp * math.cos(d2) * e3
    v_pca = m.pca(pts)[1]
    v_pca = v_pca / np.linalg.norm(v_pca)

    print("=" * 84)
    print("PCA 原理 × 三角形高场")
    print("=" * 84)
    # ① 直接: Σh_i²(v) 在 v=v_pca 时最小?
    def sum_h2(vv):
        return float(np.sum(np.cross(P, vv / np.linalg.norm(vv)) ** 2))
    print(f"  Σh_i²(v_pca) = {sum_h2(v_pca):.3f}  (应最小)")
    # 扫描若干方向对比
    rng = np.random.default_rng(0)
    dirs = rng.standard_normal((50, 3))
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    vals = [sum_h2(vv) for vv in dirs]
    print(f"  随机 50 方向: min Σh² = {min(vals):.3f} (角 {angle(dirs[np.argmin(vals)], v_pca):.1f}° 距PCA)")
    print(f"              max Σh² = {max(vals):.3f}")
    print()

    # ② 但: Σh_i²(v) = Σ|p_i×v|² 解析 = v^T(Σ(|p_i|²I - p_i p_i^T))v
    #    最小化 = 协方差主轴? 检查: |p_i×v|² = |p_i|² - (p_i·v)²
    #    Σh² = Σ|p_i|² - Σ(p_i·v)² = const - v^T(P^T P)v
    #    最小化 Σh² ⟺ 最大化 v^T(P^T P)v ⟺ 最大特征向量 = PCA 主轴 ✓ (解析确认)
    #    所以高场 Σh_i² 的最小方向 ≡ PCA 主轴 (无需协方差分解, 但等价!)
    print("解析: Σh_i²(v) = Σ|p_i|² − vᵀ(PᵀP)v → 最小方向 = PᵀP 最大特征向量 = PCA")
    print("  → 高场最小化 ≡ PCA (数学等价, 不是新信息)")
    print()

    # ③ 三角形途径: h_i 由三边 (a,b,2c) 海伦公式, a,b 由概率反演(双解)
    #    但反演只有 |cosθ_i| (无符号) → 只能得 |p_i×v| 无符号 → 高场平方不受影响!
    #    Σh_i² 用 |cosθ| 可算: h_i² = |p_i|²(1-cos²θ_i) → cos²θ_i 由 s 反演唯一确定!
    #    这给出 Σh_i²(v) 的值域判断 → 但 v 仍未知(要扫描找最小)
    print("三角形途径: cos²θ_i 由概率唯一确定(无符号问题!) → h_i² = |p_i|²(1-cos²θ_i)")
    print("  → Σh_i²(v) 作为 v 的函数可计算 → 扫描最小化 = PCA 主轴 (无协方差分解)")
    print("  → 但这是'用三角形高场重做 PCA', 不是'替代 PCA 判据'")
    print()

    # ④ 真正的问题: 终止条件 |Δ| 需要 v 与 d 的关系; 高场只给"主轴方向"
    #    若高场能定位主轴 v_recon ≈ v_pca, 则可用 v_recon 替代 v!
    #    测试: 最小 Σh² 的方向 vs v_pca
    best_v, best_val = None, 1e18
    for vv in dirs:
        val = sum_h2(vv)
        if val < best_val:
            best_val, best_v = val, vv
    print(f"  扫描最小 Σh² 方向: {np.round(best_v,3)}  vs PCA {np.round(v_pca,3)}")
    print(f"  夹角 = {angle(best_v, v_pca):.2f}°")
    print()
    print("结论: 若夹角≈0 → 三角形高场可定位主轴(等价PCA); 但 50 方向太粗, 需细扫")
    print("DONE")


if __name__ == "__main__":
    main()
