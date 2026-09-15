# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
全局规律拟合 (gaji.py 扩展):
  对大量样本, 拟合 角度 angle 与 概率组合 的关系, 找出可替代角度终止条件的条件式
  候选概率组合 (全部不依赖 PCA):
    d34 = (p4-p3)×100   [0,1格点总概率 - 理论格点总概率]
    d12 = (p2-p1)×100   [(0,1)焦点对 - 理论焦点对]
    d24 = (p4-p2)×100, d13 = (p3-p1)×100
  拟合形式: angle ≈ a·d + b  (分段: 低角区/高角区, 因概率有对称性)
"""
import math
import numpy as np

import data_driven_axis as dd
import n2sjy2 as n2
import nsjy_algorithms as m


def angle(u, v):
    return math.degrees(math.acos(np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1)))


def batch_probability(points, F1, F2, a):
    P = np.asarray(points, float)
    d1 = np.linalg.norm(P - np.asarray(F1, float), axis=1)
    d2 = np.linalg.norm(P - np.asarray(F2, float), axis=1)
    delta = d1 + d2 - 2.0 * a
    return np.exp(-np.abs(delta) / (2.0 * a))


def run_trial_fast(seed, n_points=200):
    rng = np.random.default_rng(seed)
    u = rng.normal(0, 1, (n_points, 3))
    r = rng.random(n_points) ** (1 / 3)
    points = (u / np.linalg.norm(u, axis=1, keepdims=True)) * r[:, None]
    rp = np.mean(np.linalg.norm(points, axis=1))
    d2 = math.asin(math.cos(30 * math.pi / 180) / math.pi)
    a = rp * (1 + math.sin(d2))
    _, r45, r30, _ = n2.yzqx(points, rp)
    t1, t2 = n2.compute_taus()
    t11 = t22 = complex(0, 1)
    pt1, p1, p2, sigm = dd.fast_probs(r30, r45, t1, t2, a)
    pt2, p1b, p2b, sigm2 = dd.fast_probs(r30, r45, t11, t22, a)
    F1, F2 = m.extract_foci(points, p1, sigm[1], p2, sigm[2])
    F11, F22 = m.extract_foci(points, p1b, sigm2[1], p2b, sigm2[2])
    s1 = batch_probability(points, F1, F2, a)[0]
    s2 = batch_probability(points, F11, F22, a)[0]
    s3 = pt1[0]
    s4 = pt2[0]
    v3 = (F1 - F2) / np.linalg.norm(F1 - F2)
    v31 = (F11 - F22) / np.linalg.norm(F11 - F22)
    ang = angle(v3, v31)
    return ang, s1, s2, s3, s4


def main():
    N = 4000
    rows = []
    print(f"采集 {N} 样本...", flush=True)
    for seed in range(N):
        rows.append(run_trial_fast(seed))
    A = np.array(rows)
    ang = A[:, 0]
    p1, p2, p3, p4 = A[:, 1], A[:, 2], A[:, 3], A[:, 4]

    combos = {
        "d34=(p4-p3)×100": (p4 - p3) * 100,
        "d12=(p2-p1)×100": (p2 - p1) * 100,
        "d24=(p4-p2)×100": (p4 - p2) * 100,
        "d13=(p3-p1)×100": (p3 - p1) * 100,
        "d14=(p4-p1)×100": (p4 - p1) * 100,
        "d23=(p3-p2)×100": (p3 - p2) * 100,
    }
    print("=" * 100)
    print("全局拟合: angle = a·d + b   (angle = ∠(理论焦点方向, (0,1)焦点方向))")
    print("=" * 100)
    print(f"{'组合':20s} {'corr(ang,d)':>12s} {'a(斜率)':>9s} {'b(截距)':>9s} "
          f"{'|残差|均值°':>10s} {'|残差|P90°':>10s}")
    best = None
    for name, d in combos.items():
        # 去除奇异
        mask = np.isfinite(d)
        corr = np.corrcoef(ang[mask], d[mask])[0, 1]
        a_, b_ = np.polyfit(d[mask], ang[mask], 1)
        resid = ang[mask] - (a_ * d[mask] + b_)
        rmean = np.abs(resid).mean()
        rp90 = np.percentile(np.abs(resid), 90)
        print(f"{name:20s} {corr:+12.3f} {a_:9.4f} {b_:9.3f} {rmean:10.2f} {rp90:10.2f}")
        if best is None or rmean < best[0]:
            best = (rmean, name, a_, b_)
    print()
    print(f"最佳: {best[1]}  残差均值 {best[0]:.2f}°")

    # 分段拟合 (用户 glxzf 区间): 低角 0-90 与高角 90-180
    print()
    print("=" * 100)
    print("分段拟合 (对应 glxzf 的区间划分)")
    print("=" * 100)
    for name in ["d34=(p4-p3)×100", "d12=(p2-p1)×100"]:
        d = combos[name]
        print(f"  {name}:")
        for lo, hi, tag in [(0, 90, "低角 0-90"), (90, 180, "高角 90-180")]:
            mask = (ang > lo) & (ang < hi) & np.isfinite(d)
            if mask.sum() < 10:
                continue
            corr = np.corrcoef(ang[mask], d[mask])[0, 1]
            a_, b_ = np.polyfit(d[mask], ang[mask], 1)
            resid = ang[mask] - (a_ * d[mask] + b_)
            print(f"    {tag}: corr={corr:+.3f} angle={a_:.3f}·d{b_:+.3f} "
                  f"|残差|均值={np.abs(resid).mean():.2f}° n={mask.sum()}")
    print()
    print("注: 若某组合残差小且分段稳定 → 用 d 直接判 angle 区间, 替代角度计算")
    print("DONE")


if __name__ == "__main__":
    main()
