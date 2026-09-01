# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
验证用户发现: 概率条件 (s0~s5 = BatchProbability(v_i·c, v_i·-c)) 与角度夹角的强关联
  目标: 找到概率条件关系, 替代终止条件
        if ((angleDeg+angleDeg1)*0.5 - min(angleDeg,angleDeg1)) <= 8  → 完全脱离 PCA

复现 n2sjy2.cs Start() 的 s0~s5:
  v  = PCA 主轴;      v1 = (F1-F2).norm   (理论 taus 焦点方向)
  v2 = (f1-f2).norm   (理论 taus 拟合焦点) v3 = (F01-F02).norm ((0,1) 焦点方向)
  v4 = (f01-f02).norm ((0,1) 拟合焦点)     v5 = (f1z-f2z).norm (当前迭代焦点方向)
  s_i = BatchProbability(v_i·c, v_i·-c, points, a)[0]  (第一个点的概率)

角度: angleDeg = ∠((f1-f2), v)  [依赖 PCA]
      angleDeg1 = ∠((f1-f2), (F1-F2))  [不依赖 PCA]
      |Δ| = |angleDeg - angleDeg1|
假设: |s_i - s_j| (×100) 与夹角差强相关 → 可用概率差替代角度差
"""
import json
import math

import numpy as np

import data_driven_axis as dd
import n2sjy2 as n2
import nsjy_algorithms as m

PYJSON = r"C:\Users\23128\My project (2)\Assets\Resources\pyjson.json"


def angle(u, v):
    un = np.linalg.norm(u)
    vn = np.linalg.norm(v)
    if un < 1e-12 or vn < 1e-12:
        return 0.0
    return math.degrees(math.acos(np.clip(np.dot(u, v) / (un * vn), -1, 1)))


def batch_probability(points, F1, F2, a):
    """BatchProbability: exp(-|d1+d2-2a|/(2a))  (向量化)"""
    P = np.asarray(points, float)
    d1 = np.linalg.norm(P - np.asarray(F1, float), axis=1)
    d2 = np.linalg.norm(P - np.asarray(F2, float), axis=1)
    delta = d1 + d2 - 2.0 * a
    return np.exp(-np.abs(delta) / (2.0 * a))


def setup_from_points(points):
    pts = [np.array(p, float) for p in points]
    t1t, t2t = n2.compute_taus()
    rp = n2.compute_rp(pts)
    d2 = math.asin(math.cos(math.radians(30)) / math.pi)
    a = rp * (1 + math.sin(d2))
    e3 = math.cos(d2) * 2 / (1 + math.sin(d2))
    h = rp * math.cos(d2)
    c = h * e3
    _, r45, r30, _ = n2.yzqx(pts, rp)
    # 理论 taus
    pt, p1, p2, sig = dd.fast_probs(r30, r45, t1t, t2t, a)
    F1, F2 = m.extract_foci(pts, p1, sig[1], p2, sig[2])
    # (0,1)
    t_ = complex(0, 1)
    pt01, p101, p201, sig01 = dd.fast_probs(r30, r45, t_, t_, a)
    F01, F02 = m.extract_foci(pts, p101, sig01[1], p201, sig01[2])
    v = m.pca(pts)[1]
    v = v / np.linalg.norm(v)
    # 当前方向 (模拟收敛后的 f1z-f2z ≈ 用 fit 后的 f1-f2)
    f1, f2 = m.extract_foci(pts, pt, sig[1], pt, sig[2])  # 占位
    return pts, a, c, v, F1, F2, F01, F02


def analyze(name, points, v5):
    """points: (N,3) 数组; v5: 当前焦点方向 (N=200 时的方向)"""
    pts, a, c, v, F1, F2, F01, F02 = setup_from_points(points)
    v1 = np.array(F1, float) - np.array(F2, float)
    v1 = v1 / np.linalg.norm(v1)
    v3 = np.array(F01, float) - np.array(F02, float)
    v3 = v3 / np.linalg.norm(v3)
    v5 = np.asarray(v5, float)
    v5 = v5 / np.linalg.norm(v5)

    dirs = {"v_pca": v, "v1(thF)": v1, "v3(iF)": v3, "v5(cur)": v5}
    s = {}
    for k, dvec in dirs.items():
        s[k] = batch_probability(points, dvec * c, -dvec * c, a)[0]

    # 角度: v5 与各方向夹角
    angs = {k: angle(v5, dvec) for k, dvec in dirs.items()}
    # angleDeg = ∠(v5, v)  angleDeg1 = ∠(v5, v1)
    angle_deg = angs["v_pca"]
    angle_deg1 = angs["v1(thF)"]
    delta = abs(angle_deg - angle_deg1)
    cond = (angle_deg + angle_deg1) * 0.5 - min(angle_deg, angle_deg1)

    print(f"  [{name}] ∠(v5,v)={angle_deg:6.1f}° ∠(v5,v1)={angle_deg1:6.1f}° "
          f"|Δ|={delta:6.1f}° cond={cond:5.1f}")
    print(f"     s0(v)={s['v_pca']:.5f} s1(thF)={s['v1(thF)']:.5f} "
          f"s3(iF)={s['v3(iF)']:.5f} s5(cur)={s['v5(cur)']:.5f}")
    print(f"     |s0-s5|×100={abs(s['v_pca']-s['v5(cur)'])*100:7.3f} "
          f"|s1-s5|×100={abs(s['v1(thF)']-s['v5(cur)'])*100:7.3f} "
          f"|s3-s5|×100={abs(s['v3(iF)']-s['v5(cur)'])*100:7.3f}")
    return dict(delta=delta, cond=cond, s0=s['v_pca'], s1=s['v1(thF)'],
                s3=s['v3(iF)'], s5=s['v5(cur)'])


def main():
    # 真实数据
    with open(PYJSON, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    real = np.array([[p["x"], p["y"], p["z"]] for p in raw], float)
    # 用真实数据跑一遍完整流程得到收敛方向 v5
    pts, a, c, v, F1, F2, F01, F02 = setup_from_points(real)
    rp = n2.compute_rp([np.array(p, float) for p in real])
    d2 = math.asin(math.cos(math.radians(30)) / math.pi)
    a2 = rp * (1 + math.sin(d2))
    _, r45, r30, _ = n2.yzqx([np.array(p, float) for p in real], rp)
    t1t, t2t = n2.compute_taus()
    pt, p1, p2, sig = dd.fast_probs(r30, r45, t1t, t2t, a2)
    f1z, f2z = m.extract_foci([np.array(p, float) for p in real], pt, sig[1], pt, sig[2])
    v5 = np.array(f1z, float) - np.array(f2z, float)

    print("=" * 90)
    print("概率条件 vs 角度条件 (用户发现验证)")
    print("=" * 90)
    results = []
    results.append(("pyjson真实200", real, v5))

    # 合成数据 (不同种子)
    for seed in range(3):
        e = dd.make_ellip(seed)
        results.append((f"ellip{seed}", np.array(e, float), v5))
    for seed in range(2):
        b = dd.make_ball(seed)
        results.append((f"ball{seed}", np.array(b, float), v5))

    rows = []
    for name, pts, v5d in results:
        r = analyze(name, pts, v5d)
        rows.append((name, r))
        print()

    print("=" * 90)
    print("汇总: 概率差×100 与 |Δ|/cond 的关系")
    print("=" * 90)
    print(f"{'name':14s} {'|Δ|°':>6s} {'cond':>5s} {'|s0-s5|×100':>11s} "
          f"{'|s1-s5|×100':>11s} {'|s3-s5|×100':>11s}")
    for name, r in rows:
        print(f"{name:14s} {r['delta']:6.1f} {r['cond']:5.1f} "
              f"{abs(r['s0']-r['s5'])*100:11.3f} {abs(r['s1']-r['s5'])*100:11.3f} "
              f"{abs(r['s3']-r['s5'])*100:11.3f}")
    print("DONE")


if __name__ == "__main__":
    main()
