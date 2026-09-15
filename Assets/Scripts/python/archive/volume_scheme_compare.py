# -*- coding: utf-8 -*-
"""
对比两种体积方案 (真实数据):
  A. 环体积 (Pappus):   V_torus = π a b · 2π ρ_c    ρ_c = 椭圆中心到轴的距离
  B. 椭球体积:          V_ellip = (4/3)π a b c      c = 管道厚度 (由 x1,x2 得到)
 参考体积 Vc: 该高度的局部锥体积 (1/3)π ρ² L
检查: 量纲是否闭合 [L^3], 比值是否稳定, 与参考体积是否同类型
"""
import json
import math

import numpy as np

from dd2_path_test import dd2, from_to_rotation, th2

PI = math.pi


def dd2_full(A, B, C):
    """返回 (C_ell, a_ell, b_ell, center, pC) —— 与 dd2 同算法, 额外给出平面坐标"""
    n = np.cross(B - A, C - A)
    nn = np.linalg.norm(n)
    if nn < 1e-30:
        return 0.0, 0.0, 0.0, np.zeros(2), np.zeros(2)
    n = n / nn
    e1 = (C - A) / (np.linalg.norm(C - A) + 1e-30)
    e2 = np.cross(n, e1)
    e2 /= (np.linalg.norm(e2) + 1e-30)
    A2 = np.zeros(2)
    B2 = np.array([np.dot(B - A, e1), np.dot(B - A, e2)])
    C2 = np.array([np.dot(C - A, e1), np.dot(C - A, e2)])
    center = (A2 + B2 + C2) / 3.0
    pts = [A2, B2, C2]
    sxx = sum((p[0] - center[0]) ** 2 for p in pts)
    syy = sum((p[1] - center[1]) ** 2 for p in pts)
    sxy = sum((p[0] - center[0]) * (p[1] - center[1]) for p in pts)
    tr = sxx + syy
    det = sxx * syy - sxy * sxy
    disc = math.sqrt(max(0.0, tr * tr - 4 * det))
    l1 = (tr + disc) / 2
    l2 = (tr - disc) / 2
    ae = math.sqrt(max(0.0, l1 / 2))
    be = math.sqrt(max(0.0, l2 / 2))
    if be < 1e-6:
        be = ae * 0.1
    Ce = PI * (3 * (ae + be) - math.sqrt((3 * ae + be) * (ae + 3 * be)))
    return Ce, ae, be, center, C2


def main():
    with open(r"C:\Users\23128\My project (2)\Assets\Resources\points.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    P = np.array([np.array(p, float) for p in raw], float)
    rp = float(np.mean(np.linalg.norm(P, axis=1)))
    d2d = 15.0
    alpha = math.radians(d2d)
    h = rp * math.cos(alpha)
    R = rp * math.sin(alpha)
    H = PI * R
    theta2 = PI / 2
    c1v, c2v = th2(theta2, H)
    v3 = (c1v - c2v) / np.linalg.norm(c1v - c2v)

    print(f"{'idx':>4s} {'a_ell':>7s} {'b_ell':>7s} {'ρ_c(质心距)':>11s} {'管道厚度c':>10s} "
          f"{'A面积':>8s} {'V_torus':>9s} {'V_ellip':>9s} {'Vc局部锥':>9s} "
          f"{'环/锥':>7s} {'椭球/锥':>8s}")
    for i in [0, 40, 80, 120, 160]:
        Pi = P[i]
        Y = float(Pi[1])
        v0, v1 = th2(theta2, Y)
        h1 = float(np.linalg.norm(v0 - v1))
        u2 = (v0 - v1) / (np.linalg.norm(v0 - v1) + 1e-30)
        M = from_to_rotation(v3, u2)
        Prot = M @ Pi
        y_norm = h1 / h
        Cc = np.array([0.0, y_norm * H, 0.0])
        c3, ae, be, center, pC = dd2_full(np.zeros(3), Cc, Prot)
        c2 = float(np.linalg.norm(Prot))
        Hcone = float(np.linalg.norm(Cc))

        # 管道厚度 c (由两个比值得到, 线性版卷绕系数)
        r1, r2 = c3 / c2, c3 / Hcone
        w1 = math.sqrt(max(0.0, r1 * r1 - 1))
        w2 = math.sqrt(max(0.0, r2 * r2 - 1))
        cthick = abs(h1 * w2 - h1 * w1) / (2 * PI)

        # 参考: 该高度局部锥体积 (相似律)
        # 半径 ρ_y = 0.5*h1, 高 L_y = y_norm*h
        rho_y = 0.5 * h1
        L_y = y_norm * h
        Vc = (1.0 / 3.0) * PI * rho_y * rho_y * L_y

        # A. Pappus 环体积: ρ_c = 椭圆中心到轴的距离 = center 在 e2 方向的分量
        rho_c = abs(center[1])
        A_ell = PI * ae * be
        V_torus = A_ell * 2 * PI * rho_c

        # B. 椭球体积
        V_ellip = (4.0 / 3.0) * PI * ae * be * cthick

        print(f"{i:4d} {ae:7.4f} {be:7.4f} {rho_c:11.5f} {cthick:10.6f} {A_ell:8.4f} "
              f"{V_torus:9.5f} {V_ellip:9.6f} {Vc:9.5f} "
              f"{V_torus/Vc:7.4f} {V_ellip/Vc:8.6f}")

    print()
    print("量纲检查:")
    print("  Pappus : [L²]·[L] = [L³] ✓  与参考锥体积(Vc)同为旋转体 -> 可直接比")
    print("  椭球   : [L²]·[L] = [L³] ✓  但 (a,b) 来自子午面椭圆的协方差,")
    print("           而 c 来自管道厚度(另一个几何量) -> 三轴不同源, 不构成真实椭球")


if __name__ == "__main__":
    main()
