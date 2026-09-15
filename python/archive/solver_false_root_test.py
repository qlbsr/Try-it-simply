# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
验证 ComplexAngleSolver 的牛顿法是否假收敛:
  C# NewtonSolve 收敛判据 = |Δθ| < 1e-10 (只看步长, 不看残差 |F|)
  而 θ=π/2 处 F 有局部极小(非零点) → 步长衰减 → 误判为根
检查: 各 α 下 F(π/2) 的真实值, 以及返回解的残差 |F|
"""
import cmath
import math

from lock_condition_test import F, find_solutions


def main():
    print("=== 1. θ=π/2 处 F 的真实值 (若显著非零 → 它是假根) ===")
    print(f"{'α(°)':>6s} {'F(π/2)':>14s} {'4atan²(πsinα)':>15s} {'cos²α':>8s} {'真根?':>7s}")
    for d2d in range(5, 90, 5):
        a = math.radians(d2d)
        v = F(complex(math.pi / 2, 0), a)
        term1 = 4 * math.atan(math.pi * math.sin(a)) ** 2
        c2 = math.cos(a) ** 2
        real_root = abs(v) < 1e-9
        print(f"{d2d:6d} {v.real:14.6f} {term1:15.6f} {c2:8.4f} {'YES' if real_root else 'NO(假根)':>7s}")

    print("\n=== 2. 返回解的残差 |F| (C# 只检查步长, 不检查残差) ===")
    print(f"{'α(°)':>6s} {'返回的实根θ2(°)':>16s} {'|F(θ2)|':>14s} {'是零点?':>9s}")
    for d2d in [5, 10, 15, 30, 55, 60, 75]:
        a = math.radians(d2d)
        sols = find_solutions(a)
        reals = [z for z in sols if abs(z.imag) < 1e-6]
        for z in reals[:2]:
            res = abs(F(z, a))
            print(f"{d2d:6d} {math.degrees(z.real):16.4f} {res:14.3e} "
                  f"{'是' if res < 1e-9 else '否(假收敛)':>9s}")

    print("\n=== 3. 真根 vs 假根: 用残差判据重新筛选 ===")
    print(f"{'α(°)':>6s} {'假根θ2=π/2?':>13s} {'真根(残差<1e-9)':>28s}")
    for d2d in range(5, 90, 5):
        a = math.radians(d2d)
        sols = find_solutions(a)
        good = [z for z in sols if abs(z.imag) < 1e-6 and abs(F(z, a)) < 1e-9]
        fake_pi2 = any(abs(z.real - math.pi / 2) < 1e-6 for z in sols if abs(z.imag) < 1e-6)
        gs = ", ".join(f"{math.degrees(z.real):.2f}°" for z in good) if good else "无"
        print(f"{d2d:6d} {'YES' if fake_pi2 else 'no':>13s} {gs:>28s}")


if __name__ == "__main__":
    main()
