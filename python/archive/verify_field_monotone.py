# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
用户核心假设验证: corr(s5,s0) 单调反映 ∠(d,v); ||s5-s1|| 单调反映 ∠(d,v1)
  终止条件 |∠(d,v) - ∠(d,v1)| ≤ 16 可写作两个单调函数的差:
    ∠(d,v)  = F(corr(s5, s0))      [s0 依赖 PCA]
    ∠(d,v1) = G(||s5-s1||)          [s1 无 PCA]
  但 s0 需 PCA → 用"对照场组合"近似 s0:
    想法: 主轴 v 是数据协方差最大方向, s0=P(v) 是主轴方向场
    无 PCA 近似: 用 s2 (拟合焦点 v2, 最接近 v 的对照 66.9°)?
  替代条件测试: |F(corr(s5,s2)) - G(||s5-s1||)| ≤ 16 ?
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
    v = m.pca(pts)[1]
    v = v / np.linalg.norm(v)
    v1 = (np.array(F1, float) - np.array(F2, float)); v1 = v1 / np.linalg.norm(v1)
    v2 = (np.array(f1, float) - np.array(f2, float)); v2 = v2 / np.linalg.norm(v2)
    v3 = (np.array(F01, float) - np.array(F02, float)); v3 = v3 / np.linalg.norm(v3)

    # 固定场
    s1 = batch_probability(pts, v1 * c, -v1 * c, a)
    s2 = batch_probability(pts, v2 * c, -v2 * c, a)
    s3 = batch_probability(pts, v3 * c, -v3 * c, a)
    s0 = batch_probability(pts, v * c, -v * c, a)

    # 扫描 d 沿 v1→v
    axis = np.cross(v1, v)
    if np.linalg.norm(axis) < 1e-9:
        axis = np.array([1.0, 0, 0])
    total = angle(v1, v)
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
        # 场量
        def corr(a_, b_):
            return float(np.corrcoef(a_, b_)[0, 1])
        def l2(a_, b_):
            return float(np.linalg.norm(a_ - b_) / math.sqrt(len(a_)))
        rows.append((delta, ad, ad1, corr(s5, s0), corr(s5, s1), corr(s5, s2),
                     corr(s5, s3), l2(s5, s1), l2(s5, s2)))

    print("=" * 100)
    print("场量 vs 角度: corr(s5,sX) 与 ||s5-sX|| 是否单调对应 ∠(d,vX)?")
    print("=" * 100)
    print(f"{'rot':>5s} {'|Δ|':>5s} {'∠(d,v)':>6s} {'∠(d,v1)':>6s} | "
          f"{'corr50':>6s} {'corr51':>6s} {'corr52':>6s} {'corr53':>6s} | "
          f"{'||51||':>6s} {'||52||':>6s}")
    for k in range(41):
        rot = total * k / 40
        r = rows[k]
        if k % 4 == 0:
            print(f"{rot:5.1f} {r[0]:5.1f} {r[1]:6.1f} {r[2]:6.1f} | "
                  f"{r[3]:6.3f} {r[4]:6.3f} {r[5]:6.3f} {r[6]:6.3f} | "
                  f"{r[7]:6.3f} {r[8]:6.3f}")
    print()
    # 单调性检验: corr(s5,s0) vs ∠(d,v); corr(s5,s1) vs ∠(d,v1); ||s5-s1|| vs ∠(d,v1)
    print("=" * 100)
    print("单调性/相关性检验")
    print("=" * 100)
    D = np.array([r[0] for r in rows])
    for name, col in [("corr(s5,s0)", 3), ("corr(s5,s1)", 4), ("corr(s5,s2)", 5),
                      ("corr(s5,s3)", 6), ("||s5-s1||", 7), ("||s5-s2||", 8)]:
        vals = np.array([r[col] for r in rows])
        # 与对应角度相关: ∠(d,v) 用 col 3; ∠(d,v1) 用 col 4,7
        if col in (3,):
            c = np.corrcoef([r[1] for r in rows], vals)[0, 1]
        elif col in (4, 7):
            c = np.corrcoef([r[2] for r in rows], vals)[0, 1]
        else:
            c = np.corrcoef(D, vals)[0, 1]
        print(f"  {name:12s}: corr={c:+.3f}")
    print()
    print("若 corr(s5,s0) 与 ∠(d,v) 单调相关 → 但 s0 需 PCA!")
    print("若 ||s5-s1|| 与 ∠(d,v1) 相关 → 无 PCA 可替代 angleDeg1 ✓")
    print("关键: s0 的角色能否由无 PCA 组合替代?")
    print("DONE")


if __name__ == "__main__":
    main()
