# -*- coding: utf-8 -*-
"""
验证理解模型:
  (0,1)摆动回归: 参考 (d0, d_i), 等分线 normalize(d0+d_i) 是摆动中心
  问题: ① 反射 d0 过等分线 → "另一个边界" 是 d_i 还是 pcav v?
        ② 等分线与 pcav 差多少? (v 只是指示?)
        ③ |Δ'|≤10 (d0,d_i) 判断收敛 5/7
"""
import math

import numpy as np

import data_driven_axis as dd
import n2sjy2 as n2


def angle(u, v):
    return math.degrees(math.acos(np.clip(np.dot(u, v), -1, 1)))


def setup(pts):
    rp = n2.compute_rp(pts)
    d2 = math.asin(math.cos(math.radians(30)) / math.pi)
    a = rp * (1 + math.sin(d2))
    _, r45, r30, _ = n2.yzqx(pts, rp)
    v = n2.pca_axis(pts)
    t1t, t2t = n2.compute_taus()
    pt0, p10, p20, sig0 = dd.fast_probs(r30, r45, t1t, t2t, a)
    F10, F20 = n2.extract_foci(pts, p10, sig0[1], p20, sig0[2])
    d0 = np.array(F10, float) - np.array(F20, float)
    d0 = d0 / np.linalg.norm(d0)
    pti, p1i, p2i, sigi = dd.fast_probs(r30, r45, complex(0, 1), complex(0, 1), a)
    F1i, F2i = n2.extract_foci(pts, p1i, sigi[1], p2i, sigi[2])
    d_i = np.array(F1i, float) - np.array(F2i, float)
    d_i = d_i / np.linalg.norm(d_i)
    return v, d0, d_i


def reflect_across(d, n):
    n = n / np.linalg.norm(n)
    return 2 * np.dot(d, n) * n - d


def main():
    datasets = {"cube0": dd.make_cube(10), "cube1": dd.make_cube(11),
                "ball0": dd.make_ball(0), "ball1": dd.make_ball(1),
                "ellip0": dd.make_ellip(0), "ellip1": dd.make_ellip(1)}
    rng = np.random.default_rng(1000 + 5)
    datasets["gauss5"] = [rng.standard_normal(3) * (0.5 + 2 * rng.random()) for _ in range(200)]

    print("=" * 110)
    print("理解模型验证: 扇区(d0,d_i) | 等分线 | 反射d0→另一个边界 | v仅指示?")
    print("=" * 110)
    print(f"{'name':8s} {'∠(d0,di)':>8s} {'∠(等分线,反射)':>12s} {'∠(反射,di)':>9s} "
          f"{'∠(等分线,v)':>10s} {'∠(d0,v)':>8s}")
    for name, pts in datasets.items():
        v, d0, d_i = setup(pts)
        bis = d0 + d_i
        bis = bis / np.linalg.norm(bis)
        refl = reflect_across(d0, bis)
        refl = refl / np.linalg.norm(refl)
        print(f"{name:8s} {angle(d0, d_i):8.1f} {angle(bis, refl):12.1f} "
              f"{angle(refl, d_i):9.1f} {angle(bis, v):10.1f} {angle(d0, v):8.1f}", flush=True)
    print()
    print("结论判定:")
    print("  ① 反射 d0 过等分线得到的'另一个边界' 应 ≈ d_i (∠(反射,di)→0), 而不是 pcav")
    print("  ② ∠(等分线,v) = 数据等分线与 PCA 的差距 → v 只能当指示, 不是被求的边界")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
