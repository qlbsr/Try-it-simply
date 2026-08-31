# -*- coding: utf-8 -*-
"""
验证趋势机理:
  1) |Δ| = |angle(d,v)-angle(d,d0)| 是否等价于 |d·(v-d0)| (几何V漏斗)
  2) 理想先决条件: angle(d0, v_pca) 每数据集多大 (≈0 则条件秒达成)
  3) 等角大圆: d* = normalize(v±d0) 处 |Δ| 是否≈0
"""
import math

import numpy as np

import data_driven_axis as dd
import n2sjy2 as n2


def angle(u, v):
    return math.degrees(math.acos(np.clip(np.dot(u, v), -1, 1)))


def main():
    datasets = {"cube0": dd.make_cube(10), "cube1": dd.make_cube(11),
                "ball0": dd.make_ball(0), "ball1": dd.make_ball(1),
                "ellip0": dd.make_ellip(0), "ellip1": dd.make_ellip(1)}

    print("=" * 100)
    print("1) 理想先决条件: angle(d0, v_pca)  (d0=初始焦点方向, v=PCA轴)")
    print("=" * 100)
    rows = []
    for name, pts in datasets.items():
        rp = n2.compute_rp(pts)
        d2 = math.asin(math.cos(math.radians(30)) / math.pi)
        a = rp * (1 + math.sin(d2))
        _, r45, r30, _ = n2.yzqx(pts, rp)
        t1t, t2t = n2.compute_taus()
        pt0, p10, p20, sig0 = dd.fast_probs(r30, r45, t1t, t2t, a)
        F10, F20 = n2.extract_foci(pts, p10, sig0[1], p20, sig0[2])
        d0 = np.array(F10, float) - np.array(F20, float)
        d0 = d0 / np.linalg.norm(d0)
        v = n2.pca_axis(pts)
        gap = angle(d0, v)
        rows.append((name, d0, v, gap))

        # 2) |Δ| 与 |d·(v-d0)| 的等价性: 随机采样 d
        nrm = v - d0
        nrm = nrm / np.linalg.norm(nrm)
        rng = np.random.default_rng(1)
        D = rng.standard_normal((400, 3))
        D /= np.linalg.norm(D, axis=1, keepdims=True)
        dDelta = np.array([abs(angle(d, v) - angle(d, d0)) for d in D])
        proj = np.abs(D @ nrm)
        corr = float(np.corrcoef(dDelta, proj)[0, 1])
        # 3) 等角大圆上两点
        dA = (v + d0) / np.linalg.norm(v + d0)
        dB = (v - d0) / np.linalg.norm(v - d0)
        dA_delta = abs(angle(dA, v) - angle(dA, d0))
        dB_delta = abs(angle(dB, v) - angle(dB, d0))
        print(f"{name:8s} angle(d0,v)={gap:6.1f}°   "
              f"corr(|Δ|, |d·(v-d0)|)={corr:+.3f}   "
              f"|Δ|@d*=normalize(v+d0)={dA_delta:5.2f}°  normalize(v-d0)={dB_delta:5.2f}°")

    print()
    print("=" * 100)
    print("2) 理想情况推演: 若 d0≈v (angle(d0,v)→0), 则对任意 d: |Δ|≈0 → 条件秒达成")
    print("   现实: angle(d0,v)=38~86°, 条件需要显式搜索 (nsjy4) 或部分由趋势下滑 (n2sjy2)")
    print("=" * 100)
    gaps = [r[3] for r in rows]
    print(f"angle(d0,v) 分布: min={min(gaps):.1f}°  mean={np.mean(gaps):.1f}°  "
          f"max={max(gaps):.1f}°")
    print("DONE")


if __name__ == "__main__":
    main()
