# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
验证: 自适应 γ 的软最小距离 (C# SoftMinDistance 的 Python 等价)
  1. softmin 与硬 min 的关系: softmin ≥ minDist, γ→0 时收敛到硬 min, γ→∞ 时退化
  2. 自适应 γ: 尺度 = 距离中位数 (或外部 σ), γ = clamp(γfrac·scale, 1e-6·scale, scale)
  3. 关键验证: 硬 min 在 (t1,t2) 上是分段常数 → 梯度≈0 (LM 卡死根源);
     softmin 连续 → 数值梯度非零 → LM/梯度法可用
  4. 验证 C# 表达式等价: minDist - γ·ln(Σ exp(-(d-minDist)/γ))
"""
import math
import numpy as np

import data_driven_axis as dd
import n2sjy2 as n2
import nsjy_algorithms as m


def soft_min_distance(z, lattice, gamma_frac=0.1, scale=-1.0):
    """C# SoftMinDistance 的 Python 等价 (含自适应 γ, 尺度=最近格点间隙)"""
    z = complex(z)
    dists = np.array([abs(z - lam) for lam in lattice], float)
    min_dist = dists.min()
    if scale <= 0:
        sorted_d = np.sort(dists)
        scale = max(sorted_d[1] - sorted_d[0], 1e-12) if len(sorted_d) > 1 else 1e-12
    gamma = max(1e-6 * scale, min(gamma_frac * scale, scale))
    # 数值稳定: 减去 minDist
    sum_exp = np.sum(np.exp(-(dists - min_dist) / gamma))
    return min_dist - gamma * math.log(sum_exp), gamma


def hard_min(z, lattice):
    return min(abs(z - lam) for lam in lattice)


def lattice_for(tau):
    mg = np.arange(-20, 21)
    MM, NN = np.meshgrid(mg, mg)
    return (MM + NN * tau).ravel()


def main():
    # ---- 1. 单点行为: softmin vs hardmin, γ 扫描 ----
    print("=" * 72)
    print("1. 单点: softmin vs hardmin, 固定 γ 扫描 (z=0.37+0.83j, τ=0.3+1.2i)")
    print("=" * 72)
    z = 0.37 + 0.83j
    lat = lattice_for(0.3 + 1.2j)
    hd = hard_min(z, lat)
    print(f"   硬 min = {hd:.6f}")
    for g in [1e-4, 1e-3, 0.01, 0.1, 0.5, 1.0, 5.0]:
        sm = soft_min_distance(z, lat, gamma_frac=g)[0]
        print(f"   γ={g:7.4f}: softmin={sm:.6f}  (偏差 {sm-hd:+.6f})")
    print("   → γ→0: softmin→硬min; γ→∞: 偏差增大(退化)")

    # ---- 2. 自适应 γ 的尺度匹配 ----
    print()
    print("=" * 72)
    print("2. 自适应 γ: 尺度=最近格点间隙(≈格点间距), γ=0.1·scale")
    print("=" * 72)
    sm_a, g_a = soft_min_distance(z, lat)
    sorted_d = np.sort([abs(z - lam) for lam in lat])
    gap = sorted_d[1] - sorted_d[0]
    print(f"   最近间隙={gap:.4f} 自适应γ={g_a:.4f} (0.1×{gap:.4f}={0.1*gap:.4f})")
    print(f"   自适应softmin={sm_a:.6f} (vs 硬min {hd:.6f}, 偏差 {sm_a-hd:+.6f})")

    # ---- 3. 关键: (t1,t2) 平面上硬/软梯度的连续性 ----
    print()
    print("=" * 72)
    print("3. (t1,t2) 平面上目标(焦点方向角差)的梯度: 硬min vs softmin")
    print("   硬min 分段常数 → 数值梯度≈0 → LM 卡死; softmin 应给出非零梯度")
    print("=" * 72)
    # 构造一个简单的平滑目标: 距离之和 Σ d_i(t1) (r30 通道)
    pts = dd.make_ellip(0)
    rp = n2.compute_rp(pts)
    d2 = math.asin(math.cos(math.radians(30)) / math.pi)
    a = rp * (1 + math.sin(d2))
    _, r45, r30, _ = n2.yzqx(pts, rp)
    r30a = np.asarray(r30, complex)[:40]

    def sum_dist(t1, use_soft):
        lat = lattice_for(t1)
        f = soft_min_distance if use_soft else lambda zz, L: (hard_min(zz, L), 0.0)
        return sum(f(z_, lat)[0] for z_ in r30a)

    t0 = 0.4 + 1.2j
    h = 1e-4
    for use_soft in (False, True):
        label = "softmin(自适应γ)" if use_soft else "硬min"
        g_re = (sum_dist(t0 + h, use_soft) - sum_dist(t0 - h, use_soft)) / (2 * h)
        g_im = (sum_dist(t0 + 1j * h, use_soft) - sum_dist(t0 - 1j * h, use_soft)) / (2 * h)
        print(f"   {label:16s}: ∂/∂Re={g_re:+.4f}  ∂/∂Im={g_im:+.4f}  "
              f"|∇|={math.hypot(g_re, g_im):.4f}")

    # ---- 4. 沿一条线的光滑性 ----
    print()
    print("=" * 72)
    print("4. 沿 Re(t1) 扫一条线 (Im=1.2 固定): 目标是否连续")
    print("=" * 72)
    res = []
    for re_ in np.linspace(-0.5, 0.5, 21):
        t = re_ + 1.2j
        res.append((re_, sum_dist(t, True), sum_dist(t, False)))
    for re_, sm_, hd_ in res[::4]:
        print(f"   Re={re_:+.2f}: softmin={sm_:.4f}  硬min={hd_:.4f}")
    jumps = sum(1 for i in range(1, len(res)) if abs(res[i][2] - res[i-1][2]) > 1e-6)
    print(f"   硬min 跳变次数(21点): {jumps}; softmin 连续 → 梯度信息可用")
    print("DONE")


if __name__ == "__main__":
    main()
