# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
验证命题: 三角形(原点O, 环心C, 点P) 的内切椭圆(马丁/Steiner)周长 == 该点跌落原点的路径长度?
忠实复现 dd2: 投影到平面 → 3点协方差(未除N) → 特征值 λ → 半轴 √(λ/2) → 拉马努金周长
候选"路径长度": |P|, 三角形周长, 半周长, 内切圆周长, 2π·质心距, h1(通道宽), 2π·ρ(环周长)
"""
import json
import math

import numpy as np

from forward_sjy import v3_from_angles


def from_to_rotation(a, b):
    """Rodrigues: 把 a 转到 b 的四元数/矩阵 (对应 Unity FromToRotation)"""
    a = a / (np.linalg.norm(a) + 1e-30)
    b = b / (np.linalg.norm(b) + 1e-30)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = np.linalg.norm(v)
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        # 180°: 取任一垂直轴
        perp = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
        ax = np.cross(a, perp)
        ax /= np.linalg.norm(ax)
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        return np.eye(3) + 2 * K @ K
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / (s * s))


def th2(theta, H):
    s, c = math.sin(theta), math.cos(theta)
    tanX = -H / s
    c2 = np.array([math.atan(tanX), H, tanX * c])
    tanX_low = H / s
    c1 = np.array([math.atan(tanX_low), H, tanX_low * c])
    return c1, c2


def dd2(A, B, C):
    """C# sjy.dd2 忠实: 3点 → 投影平面 → 协方差特征值 → 半轴 → 拉马努金周长"""
    n = np.cross(B - A, C - A)
    nn = np.linalg.norm(n)
    if nn < 1e-30:
        return 0.0, 0.0, 0.0
    n = n / nn
    e1 = (C - A) / (np.linalg.norm(C - A) + 1e-30)
    e2 = np.cross(n, e1)
    e2 /= (np.linalg.norm(e2) + 1e-30)
    A2 = np.array([0.0, 0.0])
    B2 = np.array([np.dot(B - A, e1), np.dot(B - A, e2)])
    C2 = np.array([np.dot(C - A, e1), np.dot(C - A, e2)])
    center = (A2 + B2 + C2) / 3.0
    pts = [A2, B2, C2]
    sxx = sum((p[0] - center[0]) ** 2 for p in pts)
    syy = sum((p[1] - center[1]) ** 2 for p in pts)
    sxy = sum((p[0] - center[0]) * (p[1] - center[1]) for p in pts)
    trace = sxx + syy
    det = sxx * syy - sxy * sxy
    disc = math.sqrt(max(0.0, trace * trace - 4 * det))
    l1 = (trace + disc) / 2
    l2 = (trace - disc) / 2
    a_ell = math.sqrt(max(0.0, l1 / 2))
    b_ell = math.sqrt(max(0.0, l2 / 2))
    if b_ell < 1e-6:
        b_ell = a_ell * 0.1
    C_ell = math.pi * (3 * (a_ell + b_ell)
                       - math.sqrt((3 * a_ell + b_ell) * (a_ell + 3 * b_ell)))
    return C_ell, a_ell, b_ell


def main():
    with open(r"C:\Users\23128\My project (2)\Assets\Resources\points.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    P = np.array([np.array(p, float) for p in raw], float)
    rp = float(np.mean(np.linalg.norm(P, axis=1)))

    for d2d in [15.0, 30.0]:
        alpha = math.radians(d2d)
        h = rp * math.cos(alpha)
        R = rp * math.sin(alpha)
        H = math.pi * R
        theta2 = math.pi / 2          # 用户坐标规范化分支 (洞口对齐坐标轴)
        c1, c2 = th2(theta2, H)
        v3 = (c1 - c2) / np.linalg.norm(c1 - c2)      # = (1,0,0)
        print(f"\n===== d2={d2d}°  h={h:.4f} R={R:.4f} H={H:.4f}  v3={v3} =====")
        print(f"  {'idx':>3s} {'|P|':>7s} {'h1':>7s} {'椭圆周长c3':>10s} {'c3/|P|':>8s} "
              f"{'c3/h1':>8s} {'2π·a_ell':>9s} {'三角周长':>9s} {'c3/半周长':>9s}")
        rows = []
        for i in [0, 20, 40, 60, 80, 100, 120, 140, 160, 180]:
            Pi = P[i]
            Y = float(Pi[1])
            v0, v1 = th2(theta2, Y)
            h1 = float(np.linalg.norm(v0 - v1))
            u2 = (v0 - v1) / (np.linalg.norm(v0 - v1) + 1e-30)
            M = from_to_rotation(v3, u2)
            Prot = M @ Pi
            Cc = np.array([0.0, Y * H, 0.0])
            c3, ae, be = dd2(np.zeros(3), Cc, Prot)
            tri_peri = (np.linalg.norm(Cc) + np.linalg.norm(Prot)
                        + np.linalg.norm(Cc - Prot))
            rows.append((i, np.linalg.norm(Prot), h1, c3, tri_peri, ae, be))
            print(f"  {i:3d} {np.linalg.norm(Prot):7.3f} {h1:7.3f} {c3:10.3f} "
                  f"{c3/(np.linalg.norm(Prot)+1e-30):8.4f} {c3/(h1+1e-30):8.4f} "
                  f"{2*math.pi*ae:9.3f} {tri_peri:9.3f} {c3/(tri_peri/2+1e-30):9.4f}")

        c3s = np.array([r[3] for r in rows])
        Pmag = np.array([r[1] for r in rows])
        h1s = np.array([r[2] for r in rows])
        tri = np.array([r[4] for r in rows])
        print(f"  相关性: corr(c3,|P|)={np.corrcoef(c3s,Pmag)[0,1]:.4f}  "
              f"corr(c3,h1)={np.corrcoef(c3s,h1s)[0,1]:.4f}  "
              f"corr(c3,三角周长)={np.corrcoef(c3s,tri)[0,1]:.4f}")
        print(f"  比值均值: c3/|P|={np.mean(c3s/Pmag):.4f}±{np.std(c3s/Pmag):.4f}  "
              f"c3/h1={np.mean(c3s/h1s):.4f}±{np.std(c3s/h1s):.4f}  "
              f"c3/(三角周长/2)={np.mean(c3s/(tri/2)):.4f}±{np.std(c3s/(tri/2)):.4f}")

    print("\n判读: 比值方差小 → 该候选量与 c3 成正比(即'路径长度'的身份)")


if __name__ == "__main__":
    main()
