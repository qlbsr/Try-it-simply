# -*- coding: utf-8 -*-
"""
分析 MapDoubleConeToLeaf 里两个角的实际数值
  x  = acos(1/sqrt(c3/c2))     c3=椭圆周长, c2=|P|(点到原点距离)
  x2 = acos(1/sqrt(c3/h1))     h1=该高度通道宽度
  a12 = x  * (pi/2) * Rad2Deg
  b1  = x2 * (pi/2) * Rad2Deg
再用 Mathf.Sin(a12), Mathf.Cos(b1) (需要弧度) 看会得到什么
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

    for d2d in [15.0, 30.0]:
        alpha = math.radians(d2d)
        h = rp * math.cos(alpha)
        R = rp * math.sin(alpha)
        H = math.pi * R
        theta2 = math.pi / 2
        c1, c2v = th2(theta2, H)
        v3 = (c1 - c2v) / np.linalg.norm(c1 - c2v)
        print(f"\n===== d2={d2d}°  h={h:.4f} H={H:.4f} =====")
        print(f"  {'idx':>3s} {'c3':>8s} {'c2=|P|':>7s} {'h1':>7s} {'c3/c2':>7s} {'c3/h1':>7s} "
              f"{'x(rad)':>7s} {'x(deg)':>7s} {'a12':>8s} {'b1':>8s} {'Sin(a12)':>9s} {'Cos(b1)':>9s}")
        for i in [0, 40, 80, 120, 160]:
            Pi = P[i]
            Y = float(Pi[1])
            v0, v1 = th2(theta2, Y)
            h1 = float(np.linalg.norm(v0 - v1))
            u2 = (v0 - v1) / (np.linalg.norm(v0 - v1) + 1e-30)
            M = from_to_rotation(v3, u2)
            Prot = M @ Pi
            Cc = np.array([0.0, Y * H, 0.0])
            c3, ae, be = dd2(np.zeros(3), Cc, Prot)
            c2 = float(np.linalg.norm(Prot))
            if c3 <= 0 or c2 <= 0 or h1 <= 0 or c3 < c2 or c3 < h1:
                print(f"  {i:3d}  c3={c3:.4f} c2={c2:.4f} h1={h1:.4f}  -> 定义域不满足, acos=NaN")
                continue
            x = math.acos(1.0 / math.sqrt(c3 / c2))
            x2 = math.acos(1.0 / math.sqrt(c3 / h1))
            a12 = x * (math.pi / 2) * RAD2DEG
            b1 = x2 * (math.pi / 2) * RAD2DEG
            print(f"  {i:3d} {c3:8.4f} {c2:7.4f} {h1:7.4f} {c3/c2:7.3f} {c3/h1:7.3f} "
                  f"{x:7.4f} {math.degrees(x):7.2f} {a12:8.2f} {b1:8.2f} "
                  f"{math.sin(a12):9.4f} {math.cos(b1):9.4f}")

    print("\n解读要点:")
    print("  · x,x2 本身是合法弧度 (0~pi/2 之间), 但 a12= x*(pi/2)*Rad2Deg 已把单位搞成")
    print("    '弧度*pi/2*度/弧度' 的混合量 (量纲=弧度·度)")
    print("  · 后面 Mathf.Sin(a12) 需要弧度输入, 却收到这个混合量 -> 数值无几何意义")
    print("  · 正确写法只需 a12 = x (弧度) 或 math.degrees(x); 且不能把结果再喂给 Sin/Cos")


if __name__ == "__main__":
    main()
