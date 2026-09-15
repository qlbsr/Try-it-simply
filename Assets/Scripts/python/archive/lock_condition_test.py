# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
按用户几何图景分析 "卡住" 条件:
  大单锥(隧道, 竖直): h=rp cosα, R=rp sinα, 开口周长相关量 π sinα
  双锥(横放): 交接环卡进大单锥开口 → theta = atan2(h/2, H), H=πR
  卡住判据: ComplexAngleSolver 方程 F(θ)=4 atan²(π sinα/sinθ)+4π²sin²α cot²θ−cos²α=0
            存在实根 (|Im θ|<1e-6) → 卡住成功
输出: 各 α 下 实根 θ2, 卡住时双锥轴 v3=(v1[0]−v1[1]).n, 方位角 fj
"""
import cmath
import math

import numpy as np


def F(theta, alpha):
    try:
        sa, ca = math.sin(alpha), math.cos(alpha)
        sinT = cmath.sin(theta)
        if abs(sinT) < 1e-300:
            return complex(1e300, 0)
        cotT = cmath.cos(theta) / sinT
        arg = math.pi * sa / sinT
        val = 4.0 * cmath.atan(arg) ** 2 + 4.0 * math.pi ** 2 * sa ** 2 * cotT ** 2 - ca ** 2
        if not (math.isfinite(val.real) and math.isfinite(val.imag)):
            return complex(float('nan'), float('nan'))
        return val
    except (OverflowError, ValueError, ZeroDivisionError):
        return complex(float('nan'), float('nan'))


def newton(guess, alpha, iters=100, tol=1e-12):
    th = guess
    for _ in range(iters):
        if abs(th) > 50:            # 发散保护
            return complex(float('nan'), float('nan'))
        f = F(th, alpha)
        df = (F(th + 1e-8, alpha) - F(th - 1e-8, alpha)) / 2e-8
        if not (math.isfinite(f.real) and math.isfinite(df.real)) or abs(df) < 1e-14:
            break
        d = f / df
        if abs(d) > math.pi:        # 步长限幅
            d = d / abs(d) * math.pi
        th -= d
        if abs(d) < tol:
            return th
    return th


def find_solutions(alpha):
    guesses = [complex(math.pi / 4, 0), complex(math.pi / 2, 0), complex(0, 1),
               complex(0, -1), complex(math.pi / 4, 1), complex(math.pi / 4, -1),
               complex(-math.pi / 4, 0), complex(-math.pi / 4, 1)]
    sols = []
    for g in guesses:
        r = newton(g, alpha)
        if math.isnan(r.real) or math.isnan(r.imag):
            continue
        if all(abs(r - s) > 1e-6 for s in sols):
            sols.append(r)
    return sols


def th2(theta, H):
    s, c = math.sin(theta), math.cos(theta)
    tanX = -H / s
    c2 = np.array([math.atan(tanX), H, tanX * c])
    tanX_low = H / s
    c1 = np.array([math.atan(tanX_low), H, tanX_low * c])
    return c1, c2


def main():
    rp = 0.9647   # points.json
    print("=== 卡住条件: 实根 θ2 是否存在, 以及卡住时双锥轴朝向 ===")
    print(f"{'d2=α(°)':>8s} {'#解':>4s} {'#实根':>5s} {'θ2(°)':>9s} {'θ_近似(°)':>10s} "
          f"{'v3=(x,0,z)':>24s} {'fj(°)':>8s} {'卡住?':>6s}")
    for d2d in range(5, 90, 5):
        alpha = math.radians(d2d)
        h = rp * math.cos(alpha)
        R = rp * math.sin(alpha)
        H = math.pi * R
        theta_ap = math.atan2(0.5 * h, H)
        sols = find_solutions(alpha)
        real_sols = [z.real for z in sols if abs(z.imag) < 1e-6]
        th = real_sols[0] if real_sols else None
        use = th if (th is not None and th != 0) else theta_ap
        c1, c2 = th2(use, H)
        d = c1 - c2
        v = d / (np.linalg.norm(d) + 1e-30)
        fj = math.degrees(math.atan2(v[2], v[0])) % 360
        ok = "YES" if (th is not None and th != 0) else "no"
        ths = f"{math.degrees(th):9.2f}" if th is not None else "     none"
        print(f"{d2d:8d} {len(sols):4d} {len(real_sols):5d} {ths} "
              f"{math.degrees(theta_ap):10.2f} ({v[0]:+.4f},0,{v[2]:+.4f}) {fj:8.2f} {ok:>6s}")

    print("\n=== 实根(复数解)分布: 检查是否有非实根 ===")
    for d2d in [10, 30, 50, 70]:
        alpha = math.radians(d2d)
        sols = find_solutions(alpha)
        print(f"  α={d2d:3d}° → {len(sols)} 个根:")
        for s in sols:
            print(f"      θ={s.real:+.6f}{s.imag:+.6f}j  |Im|={abs(s.imag):.2e} "
                  f"{'(实根→卡住)' if abs(s.imag)<1e-6 else ''}")


if __name__ == "__main__":
    main()
