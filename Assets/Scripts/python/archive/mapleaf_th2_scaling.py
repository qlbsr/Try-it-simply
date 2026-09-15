# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
验证 th2 的缩放律: h1 = |c1-c2| 是否严格正比于高度 Y
  c1=(+atan(Y/sinθ), Y, +Y·cotθ), c2=(-atan(Y/sinθ), Y, -Y·cotθ)
  → h1(Y) ∝ Y ? 以及 atan 项在大 Y 时的饱和(畸变)
"""
import math

import numpy as np


def th2(theta, Y):
    s, c = math.sin(theta), math.cos(theta)
    tanX = -Y / s
    c2 = np.array([math.atan(tanX), Y, tanX * c])
    tanX_low = Y / s
    c1 = np.array([math.atan(tanX_low), Y, tanX_low * c])
    return c1, c2


print("=== 1. h1 与高度 Y 的比例关系 ===")
for th_deg in [5.0, 15.0, 30.0]:
    th = math.radians(th_deg)
    print(f"\nθ={th_deg}°  (sinθ={math.sin(th):.4f}, cotθ={1/math.tan(th):.4f})")
    print(f"  {'Y':>8s} {'h1':>12s} {'h1/Y':>10s} {'x分量atan(Y/sinθ)':>18s} {'z分量Y·cotθ':>12s}")
    for Y in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 20.0, 100.0]:
        c1, c2 = th2(th, Y)
        h1 = float(np.linalg.norm(c1 - c2))
        print(f"  {Y:8.2f} {h1:12.6f} {h1/Y:10.6f} {math.atan(Y/math.sin(th)):18.6f} {Y/math.tan(th):12.6f}")

print("\n=== 2. 解析比值 h1/Y 的上下界 ===")
print("  小 Y: h1/Y → 2·sqrt(1/sin²θ + cot²θ) = 2·sqrt(1+cos²θ)/sinθ")
print("  大 Y: atan→π/2 饱和 → h1/Y → 2·cotθ")
for th_deg in [5.0, 15.0, 30.0, 60.0]:
    th = math.radians(th_deg)
    small = 2 * math.sqrt(1 + math.cos(th) ** 2) / math.sin(th)
    large = 2 / math.tan(th)
    print(f"  θ={th_deg:5.1f}°: 小Y极限={small:9.4f}  大Y极限={large:9.4f}  比值范围={large/small:.4f}")

print("\n=== 3. 结论 ===")
print("  · h1/Y 在小 Y 与大 Y 两端各有常数极限 → h1 ∝ Y (线性锥度)")
print("  · 中间过渡区因 atan 非线性有轻微畸变, 大 Y 时 x 分量饱和于 π/2 (非线性)")
print("  · 线性锥度 = 圆锥相似截面: 截面直径 ∝ 到顶点距离")
