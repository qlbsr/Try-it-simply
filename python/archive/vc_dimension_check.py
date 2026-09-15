# -*- coding: utf-8 -*-
"""
检验 Vc = y³·(1/3)π·H·(1/4)·h 的量纲与"基准体积"身份
  代码表达式 : (1/3)π·H·(1/4)·h  = π² R h / 12         量纲 [L²]
  正确锥体积 : (1/3)π R² h                             量纲 [L³]
  若代码表达式是"基准体积的常数倍", 比值应不随几何(NOT 依赖 R)
"""
import math

rp = 0.9647   # points.json 平均模长

print(f"{'d2(°)':>7s} {'R':>8s} {'H=πR':>8s} {'h':>8s} "
      f"{'代码基准因子':>12s} {'正确锥体积':>11s} {'比值':>8s} {'π/(4R)':>8s}")
for d2d in [5, 15, 30, 45, 60, 75, 85]:
    a = math.radians(d2d)
    h = rp * math.cos(a)
    R = rp * math.sin(a)
    H = math.pi * R
    code_factor = math.pi * H * 0.25 * h / 3.0        # π²Rh/12
    true_vol = math.pi * R * R * h / 3.0              # πR²h/3
    ratio = code_factor / true_vol
    print(f"{d2d:7d} {R:8.4f} {H:8.4f} {h:8.4f} {code_factor:12.5f} {true_vol:11.5f} "
          f"{ratio:8.4f} {math.pi/(4*R):8.4f}")

print()
print("结论:")
print("  · 代码基准因子 / 正确锥体积 = π/(4R)  —— 依赖 R(即依赖 rp 与 d2), 不是常数")
print("  · 比值随 d2 从 ~3.15 变到 ~0.84 → 代码里的那个量不是体积(量纲 [L²])")
print("  · 相似律 y³ 本身正确; 需要修的是基准因子: 用 (1/3)πR²h")
print()
print("以 H(环周长相关量) 表达的正确基准体积:")
for d2d in [15, 30, 45]:
    a = math.radians(d2d)
    h = rp * math.cos(a)
    R = rp * math.sin(a)
    H = math.pi * R
    v_from_H = (H * H / (4 * math.pi)) * h / 3.0      # = (1/3)πR²h
    print(f"  d2={d2d:3d}°: (1/3)·(H²/4π)·h = {v_from_H:.6f}  vs  (1/3)πR²h = "
          f"{math.pi*R*R*h/3:.6f}")
