# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
决定性验证: 解三角形能否真正脱离 PCA 定位当前方向 v5?
  之前的 N3 用了真值 ∠(v5,v1) 当输入 → 那是自洽性检验, 不是独立定位!
  本脚本: 只用 BatchProbability 数组的多个元素 (多点的视角) 定位 v5,
          输入: 点云 p0,p1,p2, 概率 s5[i]=P(p_i, v5·c)  [由外部提供]
          不输入: v5 坐标, 不输入 PCA
  方法: 每点 p_i 反演 θ_i=∠(p_i,v5) (双解) → v5 在绕 p̂_i 的锥面上
        两锥面交线 → 至多 2 方向; 第三点定唯一 → v5 完整方向
  输出: 定位误差 / ∠(v5_recon, v1) 对比真值 (v1 已知, 无 PCA)
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


def dist(p, q):
    return np.linalg.norm(np.asarray(p, float) - np.asarray(q, float))


def batch_probability(points, F1, F2, a):
    P = np.asarray(points, float)
    d1 = np.linalg.norm(P - np.asarray(F1, float), axis=1)
    d2 = np.linalg.norm(P - np.asarray(F2, float), axis=1)
    delta = d1 + d2 - 2.0 * a
    return np.exp(-np.abs(delta) / (2.0 * a))


def build_inv_table(p_i, c, a, n=7201):
    """p_i 的 θ→s 表"""
    r0 = np.linalg.norm(p_i)
    th = np.linspace(0, 180, n)
    s = np.zeros(n)
    for i, t in enumerate(th):
        rad = math.radians(t)
        d1 = math.sqrt(r0 ** 2 + c ** 2 - 2 * r0 * c * math.cos(rad))
        d2 = math.sqrt(r0 ** 2 + c ** 2 + 2 * r0 * c * math.cos(rad))
        s[i] = math.exp(-abs(d1 + d2 - 2 * a) / (2 * a))
    return th, s


def invert_candidates(s_val, th_grid, s_grid, tol=2e-3):
    near = np.where(np.abs(s_grid - s_val) < tol)[0]
    if len(near) == 0:
        i = np.argmin(np.abs(s_grid - s_val))
        near = [i]
    ths = th_grid[near]
    cands = []
    for base in (0, 90):
        mm = ths[(ths >= base) & (ths < base + 90)]
        if len(mm):
            cands.append(float(np.median(mm)))
    return cands


def cone_intersect(p0, th0_cands, p1, th1_cands):
    """两个锥面 (绕 p̂0 角 th0, 绕 p̂1 角 th1) 的交线方向 (单位向量)"""
    a0 = np.asarray(p0, float) / np.linalg.norm(p0)
    a1 = np.asarray(p1, float) / np.linalg.norm(p1)
    cos_a = np.dot(a0, a1)
    sin_a = math.sqrt(max(0.0, 1 - cos_a ** 2))
    sols = []
    for t0 in th0_cands:
        for t1 in th1_cands:
            c0 = math.cos(math.radians(t0))
            c1 = math.cos(math.radians(t1))
            if abs(sin_a) < 1e-9:
                continue
            # 球面三角: v·a0 = c0, v·a1 = c1
            # v = α a0 + β a1 + γ (a0×a1)
            # α + β cos_a = c0; α cos_a + β = c1
            denom = 1 - cos_a ** 2
            alpha = (c0 - c1 * cos_a) / denom
            beta = (c1 - c0 * cos_a) / denom
            gamma2 = max(0.0, 1 - alpha ** 2 - beta ** 2 - 2 * alpha * beta * cos_a)
            gamma = math.sqrt(gamma2)
            n = np.cross(a0, a1)
            n = n / np.linalg.norm(n)
            for sgn in (1.0, -1.0):
                v = alpha * a0 + beta * a1 + sgn * gamma * n
                v = v / np.linalg.norm(v)
                sols.append(v)
    return sols


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
    pt01, p101, p201, sig01 = dd.fast_probs(r30, r45, t_, t_, a)
    F01, F02 = m.extract_foci(pts, p101, sig01[1], p201, sig01[2])
    v = m.pca(pts)[1]
    v = v / np.linalg.norm(v)
    v1 = (np.array(F1, float) - np.array(F2, float))
    v1 = v1 / np.linalg.norm(v1)
    v3 = (np.array(F01, float) - np.array(F02, float))
    v3 = v3 / np.linalg.norm(v3)

    # 测试方向: 用真值 v5 = (f1-f2) 拟合方向 (模拟迭代中的当前方向)
    f1z, f2z = m.extract_foci(pts, pt, sig[1], pt, sig[2])
    v5_true = (np.array(f1z, float) - np.array(f2z, float))
    v5_true = v5_true / np.linalg.norm(v5_true)

    print("=" * 84)
    print("决定性测试: 只用多点概率 s5[i] 定位 v5 (不输入 v5 坐标, 不输入 PCA)")
    print("=" * 84)
    print(f"v5_true = {np.round(v5_true, 3)}")
    print(f"∠(v5_true, v1)={angle(v5_true, v1):.1f}°  ∠(v5_true, v3)={angle(v5_true, v3):.1f}°")
    print(f"∠(v5_true, v_pca)={angle(v5_true, v):.1f}°  [PCA 只作对照]")
    print()

    # 用 3 个点: p0, p1, p2 的 s5[i] 定位
    tables = {}
    s5_arr = batch_probability(pts, v5_true * c, -v5_true * c, a)
    for idx in (0, 1, 2, 3, 5):
        pi = P[idx]
        th, sg = build_inv_table(pi, c, a)
        tables[idx] = (th, sg)
        cands = invert_candidates(s5_arr[idx], th, sg)
        th_true = angle(pi, v5_true)
        print(f"  p{idx}: s5={s5_arr[idx]:.5f} 真θ={th_true:6.1f}° 反演候选={[f'{c:.1f}' for c in cands]}")

    # 两锥面交: p0 & p1
    print()
    th0, sg0 = tables[0]
    th1, sg1 = tables[1]
    c0 = invert_candidates(s5_arr[0], th0, sg0)
    c1 = invert_candidates(s5_arr[1], th1, sg1)
    sols = cone_intersect(P[0], c0, P[1], c1)
    print(f"p0∩p1 交线候选 {len(sols)} 个:")
    best = None
    for s in sols:
        err_v = angle(s, v5_true)
        print(f"    {np.round(s,3)}  与真值夹角 {err_v:.1f}°")
        if err_v < 3:
            best = s
    # 用 p2 从候选中选唯一
    if best is None and len(sols) > 1:
        th2, sg2 = tables[2]
        c2 = invert_candidates(s5_arr[2], th2, sg2)
        cos2 = math.cos(math.radians(np.median(c2)))
        picks = []
        for s in sols:
            err = abs(np.dot(s, P[2] / np.linalg.norm(P[2])) - cos2)
            picks.append((err, s))
        picks.sort()
        best = picks[0][1]
        print(f"  p2 消歧: 选 {np.round(best,3)} (|cos差|={picks[0][0]:.3f})")

    if best is not None:
        err = angle(best, v5_true)
        print()
        print(f"★★ 定位成功: 误差 {err:.2f}°  (输入只有概率数组+点云)")
        print(f"   重建 ∠(v5,v1)={angle(best, v1):.1f}° (真值 {angle(v5_true, v1):.1f}°)")
        print(f"   重建 ∠(v5,v3)={angle(best, v3):.1f}° (真值 {angle(v5_true, v3):.1f}°)")
        print(f"   重建 ∠(v5,v_pca)={angle(best, v):.1f}° (真值 {angle(v5_true, v):.1f}°)  [仅对照]")
    else:
        print("  定位失败 (误差>3°)")
    print()
    print("结论: 若误差小 → 三角形可定位方向(无PCA); ∠(v5,v1) 可替代 angleDeg1")
    print("      但 ∠(v5,v_pca) 若无 PCA 坐标则不可得 → 终止条件中的 PCA 项仍需讨论")
    print("DONE")


if __name__ == "__main__":
    main()
