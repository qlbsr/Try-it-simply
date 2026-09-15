# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
用户洞察验证: 三角形平面族携带 PCA 信息, 可从反演夹角无监督重建主轴
  数学: 每个点 p_i 给出 θ_i=∠(p_i, v) → p̂_i·v = ±cos θ_i (双解)
        若符号可定 → v = argmin Σ(p̂_i·v − cosθ_i)² = 协方差主轴 (等价 PCA)
  测试: ① 反演 θ_i (双解) ② 符号消歧 ③ 重建 v 对比 PCA 主轴
        ④ 替代终止条件: 用重建主轴算 ∠(d, v_recon)
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


def invert_branch(s_val, th_grid, s_grid):
    """s → θ (最近邻, 双解中取 [0,90] 分支的补角或原角)"""
    i = np.argmin(np.abs(s_grid - s_val))
    return th_grid[i]


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
    _, r45, r30, _ = n2.yzqx(pts, rp)
    t1t, t2t = n2.compute_taus()
    t_ = complex(0, 1)
    pt, p1, p2, sig = dd.fast_probs(r30, r45, t1t, t2t, a)
    F1, F2 = m.extract_foci(pts, p1, sig[1], p2, sig[2])
    v1 = (np.array(F1, float) - np.array(F2, float))
    v1 = v1 / np.linalg.norm(v1)
    f1z, f2z = m.extract_foci(pts, pt, sig[1], pt, sig[2])
    v5_true = (np.array(f1z, float) - np.array(f2z, float))
    v5_true = v5_true / np.linalg.norm(v5_true)
    v_pca = m.pca(pts)[1]
    v_pca = v_pca / np.linalg.norm(v_pca)

    # ===== 测试方向: 对目标方向 v5 (当前焦点方向), 反演全部点的 θ_i =====
    s5_arr = batch_probability(pts, v5_true * c, -v5_true * c, a)
    n = len(pts)
    # 每点独立反演表
    th_inv = np.zeros(n)
    for i in range(n):
        th, sg = build_inv_table(P[i], c, a)
        th_inv[i] = invert_branch(s5_arr[i], th, sg)   # 锐角分支 [0,90]
    print("=" * 84)
    print("从三角形平面族重建主轴 (目标: v5 当前焦点方向)")
    print("=" * 84)
    print(f"v5_true = {np.round(v5_true,3)}   v_pca = {np.round(v_pca,3)}")
    print(f"∠(v5_true, v_pca) = {angle(v5_true, v_pca):.1f}°")
    print()

    # 反演 θ_i 是锐角分支, 真实 cos = p̂_i·v5 有正有负
    Phat = P / np.linalg.norm(P, axis=1, keepdims=True)
    cos_true = Phat @ v5_true
    cos_inv = np.cos(np.radians(th_inv))   # = |p̂_i·v5| (锐角分支)
    # 符号消歧: 从 p0 出发, 用邻接传播? 更简单: 符号 = sign(p̂_i·v_pca) 近似?
    # 无监督符号: 用"多数一致" — 尝试 2^n 不可行; 用 v1 方向做初猜:
    # 三角形平面: p_i, ±v·c 共面 → 平面法向 n_i = p_i × v; v ⊥ 所有 n_i
    # 若知道 v, n_i = p_i × v; 从三角形我们只知夹角... 用符号初猜: 与 v1 同向投影
    sign_guess = np.sign(Phat @ v1 + 1e-12)
    cos_signed = cos_inv * sign_guess
    # 最小二乘重建: v = argmin Σ(p̂_i·v − cos_signed)² → 正规方程
    A = Phat.T @ Phat
    b = Phat.T @ cos_signed
    v_recon = np.linalg.solve(A + 1e-6 * np.eye(3), b)
    v_recon = v_recon / np.linalg.norm(v_recon)
    print(f"符号初猜: sign(p̂_i·v1)")
    print(f"重建 v5_recon = {np.round(v_recon,3)}  与真值夹角 {angle(v_recon, v5_true):.1f}°")
    print()

    # 尝试不同符号策略: 用 PCA 主轴符号 (对照, 非无监督)
    sign_pca = np.sign(Phat @ v_pca)
    cos_s2 = cos_inv * sign_pca
    v_r2 = np.linalg.solve(A + 1e-6 * np.eye(3), Phat.T @ cos_s2)
    v_r2 = v_r2 / np.linalg.norm(v_r2)
    print(f"符号: sign(p̂_i·v_pca) [对照] → 重建误差 {angle(v_r2, v5_true):.1f}°")
    print()

    # 无监督符号: 枚举全局符号翻转
    for sgn in (1, -1):
        v_r = np.linalg.solve(A + 1e-6 * np.eye(3), Phat.T @ (sgn * cos_inv))
        v_r = v_r / np.linalg.norm(v_r)
        print(f"符号: 全局 {sgn} → 重建 {np.round(v_r,3)} 误差 {angle(v_r, v5_true):.1f}°")

    print()
    print("=" * 84)
    print("关键: 若 cos_inv=|cos θ_i| (锐角), 则 |p̂_i·v|=cos_inv → v 在锥面族")
    print("符号消歧 = 确定每个点取 v 的哪一侧 → 需要额外信息 (邻域/主轴初猜)")
    print("结论: 三角形平面确定 '到轴的夹角' (无符号); 主轴方向需符号约束")
    print("DONE")


if __name__ == "__main__":
    main()
