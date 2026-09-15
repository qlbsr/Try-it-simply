# -*- coding: utf-8 -*-
"""
验证 vector 构造的退化:
  vector = (f1*sin(a12)*cos(b1), f2*sin(a12)*cos(b1), 0.5*h1*cos(a12)).normalized
  1) x/y 比是否恒为 f1/f2 (方位信息丢失)
  2) 自由度是否只有 1 个 (仅 t = tan(a12)*cos(b1) 变化)
  3) Sin(a12) 数值是否混乱
"""
import json
import math

import numpy as np

from dd2_path_test import dd2, from_to_rotation, th2

RAD2DEG = 180.0 / math.pi


def main():
    with open(r"C:\Users\23128\My project (2)\Assets\Resources\points.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    P = np.array([np.array(p, float) for p in raw], float)
    rp = float(np.mean(np.linalg.norm(P, axis=1)))
    d2d = 15.0
    alpha = math.radians(d2d)
    h = rp * math.cos(alpha)
    R = rp * math.sin(alpha)
    H = math.pi * R
    theta2 = math.pi / 2
    c1v, c2v = th2(theta2, H)
    v3 = (c1v - c2v) / np.linalg.norm(c1v - c2v)

    print(f"{'idx':>4s} {'f1':>8s} {'f2':>8s} {'h1':>8s} {'a12':>8s} {'b1':>8s} "
          f"{'t':>8s} {'vec':>30s} {'vec.x/vec.y':>12s} {'f1/f2':>8s}")
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
        c3, f1, f2 = dd2(np.zeros(3), Cc, Prot)
        c2 = float(np.linalg.norm(Prot))
        Hcone = float(np.linalg.norm(Cc))

        ratio1 = c3 / c2
        ratio2 = c3 / Hcone
        x1 = math.acos(1 / math.sqrt(ratio1))
        x2 = math.acos(1 / math.sqrt(ratio2))
        a12 = x1 * (math.pi / 2) * RAD2DEG
        b1 = x2 * (math.pi / 2) * RAD2DEG

        vec = np.array([f1 * math.sin(a12) * math.cos(b1),
                        f2 * math.sin(a12) * math.cos(b1),
                        0.5 * h1 * math.cos(a12)])
        vecn = vec / np.linalg.norm(vec)
        t = math.tan(a12) * math.cos(b1) if abs(math.cos(a12)) > 1e-12 else float('nan')
        ratio_xy = vecn[0] / vecn[1] if abs(vecn[1]) > 1e-12 else float('inf')
        print(f"{i:4d} {f1:8.5f} {f2:8.5f} {h1:8.5f} {a12:8.3f} {b1:8.3f} {t:8.4f} "
              f"({vecn[0]:+.4f},{vecn[1]:+.4f},{vecn[2]:+.4f}) {ratio_xy:12.6f} {f1/f2:8.6f}")

    print("\n验证要点:")
    print("  1) vec.x/vec.y 恒等于 f1/f2 → 方位(b1)完全无法改变绕轴方向")
    print("  2) 归一化后只有 t=tan(a12)cos(b1) 一个自由参数 → 参考向量只有 1 自由度")
    print("  3) a12,b1 是'弧度·度'混合量, 却喂给 sin/cos(需弧度) → 数值无几何意义")


if __name__ == "__main__":
    main()
