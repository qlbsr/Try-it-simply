# -*- coding: utf-8 -*-
"""
n2sjy2 × nsjy4 结合 (相互监督/互相拟合, PCA-free):
  1. d0 = 初始焦点方向 (理论τ, 数据驱动)
  2. d_n2 = n2sjy2 数据驱动漂移不动点 (无 PCA)
  3. d* = normalize(d0 + d_n2)  ← 双数据参考的共识等分方向 (解析解, |Δ'|=0)
  4. 迭代: d* 回喂 n2sjy2 → 新 d_n2 → 新 d* → 直到一致
  验证: 最终 d* 与 PCA 轴的夹角 (PCA 仅事后校验)
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

    print("=" * 110)
    print("单发共识: d0 → n2sjy2漂移 → d_n2 → d* = normalize(d0+d_n2)")
    print("=" * 110)
    print(f"{'name':8s} {'∠(d0,v)':>8s} {'∠(d_n2,v)':>9s} {'∠(d*,v)':>8s} "
          f"{'|Δ\'|@d*':>8s} {'n2sjy2外层':>9s}")
    results = {}
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

        # n2sjy2 数据驱动漂移 (无 PCA)
        d_n2, _, _, _, iters = dd.data_driven_axis(pts, d0)
        # 共识等分方向
        dc = d0 + d_n2
        dc = dc / np.linalg.norm(dc)
        delta_p = abs(angle(dc, d0) - angle(dc, d_n2))
        a_d0v = angle(d0, v)
        a_dn2v = angle(d_n2, v)
        a_dcv = angle(dc, v)
        results[name] = (a_d0v, a_dn2v, a_dcv)
        print(f"{name:8s} {a_d0v:8.1f} {a_dn2v:9.1f} {a_dcv:8.1f} {delta_p:8.2f} {iters:9d}")

    print()
    print("=" * 110)
    print("迭代互监督: d* 回喂 n2sjy2 → 新 d_n2 → 新 d* → 一致为止 (上限20轮)")
    print("=" * 110)
    print(f"{'name':8s} {'∠(d0,v)':>8s} {'∠(d*,v)':>8s} {'轮数':>5s} {'末轮|d*变|':>9s}")
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

        d_cons = d0
        last_chg = 999.0
        rounds = 0
        for k in range(20):
            d_n2, _, _, _, _ = dd.data_driven_axis(pts, d_cons)
            d_new = d0 + d_n2
            d_new = d_new / np.linalg.norm(d_new)
            last_chg = angle(d_new, d_cons)
            d_cons = d_new
            rounds = k + 1
            if last_chg < 0.5:
                break
        a_dcv = angle(d_cons, v)
        print(f"{name:8s} {angle(d0, v):8.1f} {a_dcv:8.1f} {rounds:5d} {last_chg:9.2f}")
    print("DONE")


if __name__ == "__main__":
    main()
