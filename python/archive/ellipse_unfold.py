# -*- coding: utf-8 -*-
"""
椭圆的标准展开 / 卷曲实现 (供 sjy 链条替换"体积比"方案)

核心对应关系
------------
圆 (k=0)                    椭圆 (0<k<1)
  w = e^z 对数展开             u = F(t,k)   第一类不完全椭圆积分 (展开)
  周期 2π                    周期 4K(k)
  半径只进对数                半轴 a,b 进模数 k = sqrt(1-b^2/a^2)

展开 : u = F(t, k) = ∫_0^t du / sqrt(1 - k^2 sin^2 u)
卷曲 : (x,y) = (a*cn(u,k), b*sn(u,k))          k=0 时退化为 (a cos u, a sin u)
周长 : C = 4 a E(k)                             k=0 时退化为 2πa
复平面: Jacobi 函数双周期 -> 格 Λ = 4K Z ⊕ 2iK' Z,   τ = i K'/K

用法 (替换 C# 里的体积比):
    ef = EllipseUnfold(a_ell, b_ell)          # 该高度的马丁椭圆半轴 (dd2 输出)
    u  = ef.unfold(t)                         # 点在椭圆上的参数角 t -> 展开坐标
    th = ef.theta(t)                          # 归一化环坐标 ∈ [0,1)
    ef.tau, ef.C, ef.k                        # 该层的模参数 / 周长 / 模数
"""
import math

import numpy as np
from scipy.special import ellipe, ellipeinc, ellipj, ellipk, ellipkinc


class EllipseUnfold:
    """一个马丁椭圆 (a,b) 的展开/卷曲算子

    注意: scipy 的 ellipk/ellipe/ellipkinc/ellipj 参数用 m = k^2,
          而 Carlsonfk.K(k) 的参数是模数 k 本身, 使用时勿混。
    """

    def __init__(self, a, b):
        a = float(a)
        b = float(b)
        if a <= 0:
            raise ValueError("长半轴 a 必须为正")
        if b > a:                       # 保证 a >= b
            a, b = b, a
        self.a = a
        self.b = b
        # 模数 = 离心率
        self.k = math.sqrt(max(0.0, 1.0 - (b * b) / (a * a)))
        self.m = self.k * self.k
        # 完全椭圆积分
        self.K = float(ellipk(self.m))              # 第一类 (实四分之一周期)
        self.Kp = float(ellipk(1.0 - self.m))       # 补模 K'(k) = K(sqrt(1-k^2))
        self.E = float(ellipe(self.m))              # 第二类
        # 周长
        self.C = 4.0 * a * self.E
        # 模参数 (纯虚, 实半轴椭圆 -> Im τ > 0 自动成立)
        self.tau = 1j * (self.Kp / self.K) if self.K > 0 else complex(0, math.inf)
        # 环的实周期
        self.period = 4.0 * self.K if self.K > 0 else math.inf

    # ---------- 展开: 椭圆 -> 直线 ----------
    def unfold(self, t):
        """椭圆参数角 t (弧度, 对应 (a cos t, b sin t)) -> 展开坐标 u"""
        return float(ellipkinc(t, self.m))

    def theta(self, t):
        """归一化环坐标 ∈ [0,1): theta = u / (4K)"""
        if not np.isfinite(self.period) or self.period == 0:
            return float('nan')
        return self.unfold(t) / self.period

    # ---------- 卷曲: 直线 -> 椭圆 ----------
    def fold_angle(self, u):
        """展开坐标 u -> 椭圆参数角 t (= Jacobi 振幅 am(u,m))"""
        sn, cn, dn, ph = ellipj(u, self.m)
        return float(ph)

    def fold_point(self, u):
        """展开坐标 u -> 椭圆上的点 (x,y) = (a cn u, b sn u)"""
        sn, cn, dn, ph = ellipj(u, self.m)
        return float(self.a * cn), float(self.b * sn)

    # ---------- 与 dd2 接口 ----------
    def summary(self):
        return dict(a=self.a, b=self.b, k=self.k, K=self.K, Kp=self.Kp,
                    E=self.E, C=self.C, tau=self.tau)


def _arc_length_numeric(a, b, t, n=20000):
    """数值弧长 (用于校验 C = 4aE(k))"""
    ts = np.linspace(0, t, n)
    integrand = np.sqrt(a * a * np.sin(ts) ** 2 + b * b * np.cos(ts) ** 2)
    return float(np.trapezoid(integrand, ts))


def self_test():
    print("=" * 78)
    print("自检 1: k=0 (圆) 应退化为 标准圆 + 对数展开")
    ef = EllipseUnfold(1.0, 1.0)
    print(f"  a=b=1 -> k={ef.k:.3e}  K={ef.K:.6f} (应=π/2={math.pi/2:.6f})  "
          f"K'={ef.Kp:.6f}  τ={ef.tau}  周长 C={ef.C:.6f} (应=2π={2*math.pi:.6f})")
    for t in [0.3, 1.0, 2.0]:
        u = ef.unfold(t)
        th = ef.theta(t)
        x, y = ef.fold_point(u)
        print(f"    t={t:.3f} -> u={u:.6f} (应≈t)  θ={th:.6f} (应≈t/2π={t/(2*math.pi):.6f})  "
              f"卷回=({x:.6f},{y:.6f}) (应≈({math.cos(t):.6f},{math.sin(t):.6f}))")

    print("=" * 78)
    print("自检 2: 一般椭圆 展开->卷曲 往返一致性")
    for a, b in [(2.0, 1.0), (1.0, 0.5), (3.0, 2.5)]:
        ef = EllipseUnfold(a, b)
        err = 0.0
        for t in np.linspace(0, 2 * math.pi, 13)[:-1]:
            u = ef.unfold(t)
            x, y = ef.fold_point(u)
            err = max(err, abs(x - a * math.cos(t)), abs(y - b * math.sin(t)))
        print(f"  a={a} b={b}: k={ef.k:.6f}  往返最大误差={err:.3e}  τ={ef.tau.real:.1f}{ef.tau.imag:+.6f}j")

    print("=" * 78)
    print("自检 3: 周长 C = 4aE(k) vs 数值积分")
    for a, b in [(2.0, 1.0), (1.0, 0.5), (1.0, 0.2), (1.0, 0.99)]:
        ef = EllipseUnfold(a, b)
        num = _arc_length_numeric(a, b, 2 * math.pi)
        print(f"  a={a} b={b}: k={ef.k:.5f}  C公式={ef.C:.8f}  数值={num:.8f}  "
              f"相对差={abs(ef.C-num)/num:.2e}")

    print("=" * 78)
    print("自检 4: 跌落过程 k: 0 -> 1  (τ 沿虚轴从 i∞ 流向 0)")
    print(f"  {'b/a':>8s} {'k=离心率':>10s} {'K':>10s} {'K′':>10s} {'Im τ':>12s} {'C/(4a)':>9s}")
    for ratio in [1.0, 0.99, 0.95, 0.9, 0.8, 0.6, 0.4, 0.2, 0.05]:
        ef = EllipseUnfold(1.0, ratio)
        print(f"  {ratio:8.3f} {ef.k:10.5f} {ef.K:10.5f} {ef.Kp:10.5f} "
              f"{ef.tau.imag:12.5f} {ef.C/4.0:9.5f}")
    print("  说明: b/a=1 (圆) -> Im τ = ∞;  b/a→0 (退化) -> Im τ → 0")


def demo_with_dd2_output():
    """演示: 给定 dd2 输出的半轴 (a_ell,b_ell) 与点的参数角, 得到环坐标"""
    print("=" * 78)
    print("演示: dd2 半轴 -> 环坐标 θ 与 该层 τ (替换体积比方案)")
    print(f"  {'a_ell':>8s} {'b_ell':>8s} {'k':>8s} {'θ(t=1rad)':>11s} {'Im τ':>10s} {'周长C':>10s}")
    for a_ell, b_ell in [(1.0, 1.0), (1.2, 1.0), (1.5, 1.0), (2.0, 1.0), (3.0, 1.0)]:
        ef = EllipseUnfold(a_ell, b_ell)
        print(f"  {a_ell:8.2f} {b_ell:8.2f} {ef.k:8.4f} {ef.theta(1.0):11.6f} "
              f"{ef.tau.imag:10.5f} {ef.C:10.5f}")


if __name__ == "__main__":
    self_test()
    demo_with_dd2_output()
