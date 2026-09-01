# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
用户对照实验: 6 条向量 → 6 个概率场 (每个点 p_i 与 ±v·c 的三角形)
  s0 = P(v)      PCA 向量 (固定)
  s5 = P(d)      迭代向量 (变化)
  s1..s4         v1..v4 固定向量 (对照)
验证:
  1. 对照有效性: s1..s4 场在迭代/扫描中是否真不变?
  2. 场形状关系: d 场与 PCA 场的关系 (差/相关/形状距离) vs |Δ|
  3. 能否用场关系替代终止条件 (|Δ|≤16)?
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


def field_shape(s):
    """概率场的形状特征 (整个分布)"""
    s = np.asarray(s, float)
    return {
        "mean": s.mean(),
        "std": s.std(),
        "skew": float(np.mean((s - s.mean()) ** 3) / (s.std() ** 3 + 1e-12)),
        "entropy": float(-np.sum(s * np.log(s + 1e-12))),
        "max": s.max(),
        "min": s.min(),
        "range": float(s.max() - s.min()),
    }


def shape_dist(sa, sb):
    """两个场的形状距离: 标准化差"""
    a = (np.asarray(sa, float) - np.mean(sa)) / (np.std(sa) + 1e-12)
    b = (np.asarray(sb, float) - np.mean(sb)) / (np.std(sb) + 1e-12)
    return float(np.linalg.norm(a - b) / math.sqrt(len(a))), float(np.corrcoef(a, b)[0, 1])


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
    f1, f2 = m.extract_foci(pts, pt, sig[1], pt, sig[2])
    pt01, p101, p201, sig01 = dd.fast_probs(r30, r45, t_, t_, a)
    F01, F02 = m.extract_foci(pts, p101, sig01[1], p201, sig01[2])
    f01, f02 = m.extract_foci(pts, pt01, sig01[1], pt01, sig01[2])
    v = m.pca(pts)[1]
    v = v / np.linalg.norm(v)
    v1 = (np.array(F1, float) - np.array(F2, float)); v1 = v1 / np.linalg.norm(v1)
    v2 = (np.array(f1, float) - np.array(f2, float)); v2 = v2 / np.linalg.norm(v2)
    v3 = (np.array(F01, float) - np.array(F02, float)); v3 = v3 / np.linalg.norm(v3)
    v4 = (np.array(f01, float) - np.array(f02, float)); v4 = v4 / np.linalg.norm(v4)

    # 固定场 (对照)
    s1 = batch_probability(pts, v1 * c, -v1 * c, a)
    s2 = batch_probability(pts, v2 * c, -v2 * c, a)
    s3 = batch_probability(pts, v3 * c, -v3 * c, a)
    s4 = batch_probability(pts, v4 * c, -v4 * c, a)
    s0 = batch_probability(pts, v * c, -v * c, a)
    refs = {"s1": s1, "s2": s2, "s3": s3, "s4": s4, "s0(pca)": s0}

    # 沿 v1→v 大圆扫描 d (模拟迭代轨迹)
    axis = np.cross(v1, v)
    if np.linalg.norm(axis) < 1e-9:
        axis = np.array([1.0, 0, 0])
    total = angle(v1, v)

    print("=" * 100)
    print("6 场对照实验: d 沿 v1→v 大圆扫描 (41 步)")
    print("=" * 100)
    rows = []
    for k in range(41):
        rot = total * k / 40
        ang = math.radians(rot)
        d = (v1 * math.cos(ang) + np.cross(axis, v1) * math.sin(ang)
             + axis * np.dot(axis, v1) * (1 - math.cos(ang)))
        d = d / np.linalg.norm(d)
        ad = angle(d, v)
        ad1 = angle(d, v1)
        delta = abs(ad - ad1)
        s5 = batch_probability(pts, d * c, -d * c, a)
        # 对照场不变性: s1..s4 与初始的差 (应≈0)
        ctrl = {k_: float(np.linalg.norm(v_ - s5)) for k_, v_ in refs.items()}
        rows.append((delta, ad, ad1, s5, dict(ctrl)))
    print(f"{'rot':>5s} {'|Δ|':>6s} | {'||s5-s0||':>9s} {'||s5-s1||':>9s} "
          f"{'||s5-s3||':>9s} {'corr(s5,s0)':>11s}")
    for k in range(41):
        rot = total * k / 40
        delta, ad, ad1, s5, ctrl = rows[k]
        if k % 4 == 0 or delta <= 16:
            d5, c50 = shape_dist(s5, refs["s0(pca)"])
            print(f"{rot:5.1f} {delta:6.1f} | {ctrl['s0(pca)']:9.3f} {ctrl['s1']:9.3f} "
                  f"{ctrl['s3']:9.3f} {c50:11.3f}")
    print()

    # 对照场是否恒定的验证: 所有 s1..s4 场不随 d 变化 (它们确实不含 d)
    print("=" * 100)
    print("对照有效性: s1..s4 场不依赖 d (由定义即恒定) → 对照成立")
    print("=" * 100)
    print("  s1..s4 只依赖 (v_i, 点云), 与 d 无关 → 迭代中恒为常数 ✓")
    print()

    # 场关系 vs |Δ|: 用 shape_dist 做替代候选
    print("=" * 100)
    print("替代候选: shape_dist(s5, s0) / shape_dist(s5, s1) 与 |Δ| 的相关性")
    print("=" * 100)
    D0, D1, DEL = [], [], []
    for delta, ad, ad1, s5, ctrl in rows:
        d0, c0 = shape_dist(s5, refs["s0(pca)"])
        d1, c1 = shape_dist(s5, refs["s1"])
        D0.append(d0)
        D1.append(d1)
        DEL.append(delta)
    D0, D1, DEL = np.array(D0), np.array(D1), np.array(DEL)
    print(f"  corr(|Δ|, shape_dist(s5,s0)) = {np.corrcoef(DEL, D0)[0,1]:+.3f}")
    print(f"  corr(|Δ|, shape_dist(s5,s1)) = {np.corrcoef(DEL, D1)[0,1]:+.3f}")
    # 原条件 |Δ|≤16 vs 场条件
    ok = DEL <= 16
    for name, D in [("shape(s5,s0)", D0), ("shape(s5,s1)", D1)]:
        best_acc, best_t = 0, 0
        for T in np.arange(0.01, 3, 0.02):
            pred = D <= T
            acc = (pred == ok).mean()
            if acc > best_acc:
                best_acc, best_t = acc, T
        print(f"  条件 {name} ≤ {best_t:.2f} ⟺ |Δ|≤16: 一致率 {best_acc*100:.0f}%")
    print()
    print("DONE")


if __name__ == "__main__":
    main()
