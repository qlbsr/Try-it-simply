# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
总结条件式: 用概率替代角度 (完全脱离 PCA)
规律 (已验证):
  R1: d12=(p2-p1)×100 与 d34=(p4-p3)×100 相关 corr 0.984, |d34/d12|≈2.83
  R2: |d34| 随 angle 单调递增: 4.31→12.30 (9.8°→170.2°), 斜率≈0.05/°
  R3: glxzf 164-180: corr(d1-d2, jd) = -0.853

条件式候选:
  C1: angle ≈ a·d34 + b   (全局线性反推)
  C2: 分段线性 (16/30/45/74 区间)
  C3: 用 d34 判断"两角差"终止: 概率版终止条件
评估: 反推角度残差 + 终止一致性
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


def trial(seed):
    rng = np.random.default_rng(seed)
    u = rng.normal(0, 1, (200, 3))
    r = rng.random(200) ** (1 / 3)
    points = (u / np.linalg.norm(u, axis=1, keepdims=True)) * r[:, None]
    rp = np.mean(np.linalg.norm(points, axis=1))
    d2 = math.asin(math.cos(30 * math.pi / 180) / math.pi)
    a = rp * (1 + math.sin(d2))
    _, r45, r30, _ = n2.yzqx(points, rp)
    t1, t2 = n2.compute_taus()
    t11 = t22 = complex(0, 1)
    pt1, p1, p2, sig = dd.fast_probs(r30, r45, t1, t2, a)
    pt2, p1b, p2b, sig2 = dd.fast_probs(r30, r45, t11, t22, a)
    F1, F2 = m.extract_foci(points, p1, sig[1], p2, sig[2])
    F11, F22 = m.extract_foci(points, p1b, sig2[1], p2b, sig2[2])
    s1 = batch_probability(points, F1, F2, a)[0]
    s2 = batch_probability(points, F11, F22, a)[0]
    s3 = pt1[0]
    s4 = pt2[0]
    v3 = (F1 - F2) / np.linalg.norm(F1 - F2)
    v31 = (F11 - F22) / np.linalg.norm(F11 - F22)
    ang = angle(v3, v31)
    return ang, s1, s2, s3, s4


def main():
    N = 6000
    rows = []
    for seed in range(N):
        rows.append(trial(seed))
    A = np.array(rows)
    ang = A[:, 0]
    p1, p2, p3, p4 = A[:, 1], A[:, 2], A[:, 3], A[:, 4]
    d34 = (p4 - p3) * 100
    d12 = (p2 - p1) * 100

    print("=" * 78)
    print("C1: 全局线性反推  angle ≈ a·d34 + b")
    print("=" * 78)
    a_, b_ = np.polyfit(d34, ang, 1)
    resid = ang - (a_ * d34 + b_)
    print(f"  angle ≈ {a_:.3f}·d34 {b_:+.2f}   corr={np.corrcoef(d34, ang)[0,1]:+.3f}")
    print(f"  残差: 均值 {np.abs(resid).mean():.2f}°  P90 {np.percentile(np.abs(resid),90):.2f}°")
    print()

    print("=" * 78)
    print("C2: 分段线性 (16/30/45/74 区间)")
    print("=" * 78)
    segs = [(0, 16), (16, 30), (30, 45), (45, 74), (74, 90),
            (90, 164), (164, 180)]
    coeffs = {}
    all_resid = []
    for lo, hi in segs:
        mk = (ang > lo) & (ang < hi) & (np.abs(d12) > 0.5) & (np.abs(d34) > 0.5)
        if mk.sum() < 20:
            continue
        aa, bb = np.polyfit(d34[mk], ang[mk], 1)
        coeffs[(lo, hi)] = (aa, bb)
        r = ang[mk] - (aa * d34[mk] + bb)
        all_resid.extend(np.abs(r))
        print(f"  ({lo:3d},{hi:3d}): angle ≈ {aa:.3f}·d34 {bb:+.2f}  "
              f"corr {np.corrcoef(d34[mk], ang[mk])[0,1]:+.3f}  n={mk.sum()}")
    print(f"  分段残差: 均值 {np.mean(all_resid):.2f}°")
    print()

    print("=" * 78)
    print("C3: 概率版终止条件评估 (替代 |Δ|≤16)")
    print("    注: angle 为 ∠(v1理论, v3(0,1)) 固定夹角, 非迭代量;")
    print("    真正终止量是迭代中 ∠(d,v) 与 ∠(d,v1) 之差, 需在迭代轨迹上验证")
    print("=" * 78)
    # 用 d34 的"符号模式"判断: d34 大 → 高角; 结合 d12 符号
    print("  规律总结:")
    print("    |d34| 随角度单调增: 4.31@10° → 12.30@170° (斜率≈0.05/°)")
    print("    同号规则: d34 与 d12 同号 (corr 0.984), d34 ≈ 2.83·d12")
    print("    glxzf 164-180: dd≈72.6±3.3 稳定, corr(d1-d2,jd)=-0.853")
    print()
    print("  建议条件式 (脱离 PCA):")
    print("    参考角估计: angle_ref = 0.05·|d34| + 10  (或分段公式)")
    print("    终止判据(概率版): |s5 - s1|·100 < T1 且 |s5 - s3|·100 < T2")
    print("      s5 = P(当前方向), s1 = P(理论焦点方向), s3 = P((0,1)焦点方向)")
    print("      T1/T2 由迭代轨迹标定 (见 analyze_iter_prob 数据: 收敛时 |s1-s5| 小)")
    print("DONE")


if __name__ == "__main__":
    main()
