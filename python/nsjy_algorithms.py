# -*- coding: utf-8 -*-
"""
nsjy.cs + n2sjy.cs 算法提取（C# → Python 移植）

来源文件:
  - nsjy.cs  : 主流程(nsjy.Start)、InverseTh4/yzqx 圆锥坐标变换、ComputeRp、ComputeTaus、
               Carlsonfk(复数椭圆积分 RF/RD/K/E/DK/NK)、ExtractFoci/FitFocus、
               FitFociByProbability/BatchProbability、pca(PCA 主轴)、
               LatticeAnalysis(格基估计/周期矩阵)、LLL 归约、ComplexLinearFit、LatticeRansac
  - n2sjy.cs : 闭环优化 RefineModuliByAxis + EvaluateResiduals + NormalizeTau + Solve4

与 C# 的已知差异(有意修正, 均用注释标出):
  1. BatchProbability 的 `-abs(delta)/2*a` 运算符优先级 bug → 按数学意图写 `-abs(delta)/(2*a)`
  2. LLL 的 static bstar 列表 bug(每次调用重复 Add) → 改为每次调用重新计算的干净实现
  3. n2sjy iter==3 重启块里 angleDeg2 误用了旧 dir 变量 → 用 dir0 计算
  4. pca() 额外返回 pc1(闭环建议直接用 pc1 而不是重建的 v3)
  5. 最近格点距离用 numpy 向量化(数学等价, 快得多)

依赖: numpy (标准库 cmath/math/itertools/json)
运行: python nsjy_algorithms.py [points.json] [--lattice] [--seed N]
      points.json: 数组 [[x,y,z],...] 或每行 "x y z"
"""
from __future__ import annotations
import cmath
import itertools
import json
import math
import sys

import numpy as np

# ======================================================================
# 0. 基础工具 (Unity Vector3 / Mathf 等价)
# ======================================================================

def v3(x: float, y: float, z: float) -> np.ndarray:
    return np.array([x, y, z], dtype=float)


def dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def sqr_magnitude(v: np.ndarray) -> float:
    return float(np.dot(v, v))


def normalized(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v.copy()


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def csqrt(z: complex) -> complex:
    """C# Complex.Sqrt 主分支"""
    return cmath.sqrt(z)


def compute_rp(points) -> float:
    """nsjy.ComputeRp: 平均半径"""
    if len(points) == 0:
        return 0.0
    return float(np.mean([np.linalg.norm(p) for p in points]))


# ======================================================================
# 1. Gamma 函数 (Lanczos, 与 nsjy.Gamma 一致)
# ======================================================================

def gamma(x: float) -> float:
    p = [676.5203681218851, -1259.1392167224028, 771.32342877765313,
         -176.61502916214059, 12.507343278686905, -0.13857109526572012,
         9.9843695780195716e-6, 1.5056327351493116e-7]
    if x < 0.5:
        return math.pi / (math.sin(math.pi * x) * gamma(1.0 - x))
    x -= 1.0
    a = 0.99999999999980993
    for i in range(len(p)):
        a += p[i] / (x + i + 1.0)
    t = x + len(p) - 0.5
    return math.sqrt(2 * math.pi) * (t ** (x + 0.5)) * math.exp(-t) * a


def elliptic_integral_k(m: float) -> complex:
    """Meta.Numerics elliptic.ellipticintegralk(m, xdefault) = RF(0, 1-m, 1)"""
    return Carlsonfk.RF(0j, 1.0 - m + 0j, 1.0 + 0j)


# ======================================================================
# 2. Carlsonfk: 复数对称椭圆积分 (nsjy.cs 933-1110 行)
# ======================================================================

class Carlsonfk:
    @staticmethod
    def K(k: complex) -> complex:
        """K(k) = RF(0, 1-k², 1)"""
        return Carlsonfk.RF(0j, 1.0 - k * k, 1.0 + 0j)

    @staticmethod
    def RF(x: complex, y: complex, z: complex) -> complex:
        x, y, z = complex(x), complex(y), complex(z)
        X = Y = Z = u = 0j
        eps = 1e-12
        for _ in range(300):
            u = (x + y + z) / 3.0
            X = (u - x) / u
            Y = (u - y) / u
            Z = (u - z) / u
            if max(abs(X), abs(Y), abs(Z)) < eps * abs(u):
                break
            sx, sy, sz = csqrt(x), csqrt(y), csqrt(z)
            lam = sx * sy + sx * sz + sy * sz
            x, y, z = (x + lam) / 4.0, (y + lam) / 4.0, (z + lam) / 4.0
        E2 = X * Y - Z * Z
        E3 = X * Y * Z
        return (1.0 + E2 * (E2 / 24.0 - E3 * 3.0 / 44.0 - 0.1) + E3 / 14.0) / csqrt(u)

    @staticmethod
    def RD(x: complex, y: complex, z: complex) -> complex:
        x, y, z = complex(x), complex(y), complex(z)
        eps = math.sqrt(2.220446049250313e-16)
        if x == y:
            sx, sy = csqrt(x), csqrt(y)
            s = 0j
            pw = 0.25
            for _ in range(300):
                if abs(sx - sy) <= 2.7 * eps * abs(sx):
                    break
                t = csqrt(sx * sy)
                sx = (sx + sy) / 2.0
                sy = t
                pw *= 2.0
                s += pw * (sx - sy) * (sx - sy)
            rf_ = math.pi / (sx + sy)
            pt = (sx + 3.0 * sy) / (4.0 * z * (sx + sy))
            pt -= s / (z * (y - z))
            return pt * rf_ * 3.0
        An = (x + y + 3.0 * z) / 5.0
        A0 = An
        q = (math.e / 4.0) ** (-1.0 / 8.0)
        fn = 1.0
        rd_sum = 0j
        for _ in range(300):
            sx, sy, sz = csqrt(x), csqrt(y), csqrt(z)
            lam = sx * sy + sx * sz + sy * sz
            rd_sum += fn / (sz * (z + lam))
            An = (An + lam) / 4.0
            x, y, z = (x + lam) / 4.0, (y + lam) / 4.0, (z + lam) / 4.0
            fn /= 4.0
            q /= 4.0
            if q.real < An.real:
                break
        X = fn * (A0 - x) / An
        Y = fn * (A0 - y) / An
        Z = -(X + Y) / 3.0
        E2 = X * Y - 6.0 * Z * Z
        E3 = (3.0 * X * Y - 8.0 * Z * Z) * Z
        E4 = 3.0 * (X * Y - Z * Z) * Z * Z
        E5 = X * Y * Z * Z * Z
        result = fn * (An ** (-3.0 / 2.0)) * (
            1.0 - 3.0 * E2 / 14.0 + E3 / 6.0 + 9.0 * E2 * E2 / 88.0
            - 3.0 * E4 / 22.0 - 9.0 * E2 * E3 / 52.0 + 3.0 * E5 / 26.0
            - E2 * E2 * E2 / 16.0 + 3.0 * E3 * E3 / 40.0
            + 3.0 * E2 * E4 / 20.0 + 45.0 * E2 * E2 * E3 / 272.0
            - 9.0 * (E3 * E4 + E2 * E5) / 68.0)
        return result + 3.0 * rd_sum

    @staticmethod
    def integrate(f, a: float, b: float, n: int) -> complex:
        """Simpson 复积分"""
        if n % 2 != 0:
            n += 1
        h = (b - a) / n
        s = f(a) + f(b)
        for i in range(1, n):
            x = a + i * h
            s += 2.0 * f(x) if i % 2 == 0 else 4.0 * f(x)
        return s * h / 3.0

    @staticmethod
    def E(k: complex) -> complex:
        """E(k) = ∫₀^{π/2} sqrt(1 - k² sin²θ) dθ"""
        return Carlsonfk.integrate(
            lambda th: csqrt(1.0 - k * k * cmath.sin(th) ** 2),
            0.0, math.pi / 2.0, 2000)

    @staticmethod
    def RD1z(k: complex) -> complex:
        Kk = Carlsonfk.K(k)
        Ek = Carlsonfk.E(k)
        return (3.0 / (k * k)) * (Kk - Ek)

    @staticmethod
    def RDz1(k: complex) -> complex:
        Ek = Carlsonfk.E(k)
        return (Ek * 3.0 / (1.0 - k * k)) - Carlsonfk.RD1z(k)

    @staticmethod
    def DK(k: complex) -> complex:
        return (Carlsonfk.E(k) - (1.0 - k * k) * Carlsonfk.K(k)) / (k * (1.0 - k * k))

    @staticmethod
    def NK(aot: complex) -> complex:
        """牛顿法反解 K(k) = aot"""
        k = 0j
        h = 1e-6
        if aot.real < 0:
            aot = csqrt(aot ** 2)
            print(" 实数为负")
        for _ in range(500):
            f = Carlsonfk.K(k) - aot
            if abs(f) < 1e-12:
                break
            df = (Carlsonfk.K(k + h) - Carlsonfk.K(k - h)) / (2.0 * h)
            if abs(df) < 1e-14:
                k += complex(0.01, 0.01)
                continue
            step = f / df
            if abs(step) > 0.2:
                step *= 0.2 / abs(step)
            k -= step
        return k


# ======================================================================
# 3. 圆锥坐标变换与模量初值 (nsjy.cs)
# ======================================================================

def inverse_th4(vs3: np.ndarray, d2_deg: float, rp: float) -> complex:
    """nsjy.InverseTh4: 3D 点 → 复平面 (u,v)"""
    d2 = math.radians(d2_deg)
    rs = math.hypot(vs3[0], vs3[1])
    zs = vs3[2]
    phis = math.atan2(vs3[1], vs3[0])
    if phis < 0:
        phis += 2 * math.pi
    r = math.hypot(rs, zs)
    sin_d2 = math.sin(d2)
    thetas = phis / sin_d2
    thetas %= 2 * math.pi
    Rs = (r / rp) ** (1.0 / sin_d2)
    return complex(Rs * math.cos(thetas), Rs * math.sin(thetas))


def yzqx(points, rp: float):
    """nsjy.yzqx: 计算 r45/r30/r74 并返回归一化的 (r30,r45) 对列表"""
    r45 = [inverse_th4(p, 45, rp) for p in points]
    r30 = [inverse_th4(p, 30, rp) for p in points]
    r74 = [inverse_th4(p, 74, rp) for p in points]

    def valid(z: complex) -> bool:
        return not (math.isnan(z.real) or math.isinf(z.real)
                    or math.isnan(z.imag) or math.isinf(z.imag))

    raw = list(zip(r30, r45))
    valid_list = [(x, y) for (x, y) in raw if valid(x) and valid(y)]
    if not valid_list:
        return valid_list, r45, r30, r74

    threshold = 1e10
    filtered = [(x, y) for (x, y) in valid_list if abs(x) < threshold and abs(y) < threshold]
    if not filtered:
        return filtered, r45, r30, r74

    max_mag = max(max(abs(x), abs(y)) for (x, y) in filtered)
    if max_mag > 0:
        return [(x / max_mag, y / max_mag) for (x, y) in filtered], r45, r30, r74
    return filtered, r45, r30, r74


def compute_taus():
    """nsjy.ComputeTaus: tau1 = i·K(i√3)/K(2), tau2 = i·K(i)/K(√2)"""
    K2 = Carlsonfk.K(2.0)
    Kprime2 = Carlsonfk.K(csqrt(1.0 - 2.0 * 2.0))          # K(i√3)
    tau1 = 1j * Kprime2 / K2

    Ksqrt2 = Carlsonfk.K(math.sqrt(2))
    KprimeSqrt2 = Carlsonfk.K(csqrt(1.0 - 2.0))            # K(i)
    tau2 = 1j * KprimeSqrt2 / Ksqrt2

    if tau1.imag < 0:
        tau1 = -tau1
    if tau2.imag < 0:
        tau2 = -tau2
    return tau1, tau2


# ======================================================================
# 4. PCA (nsjy.pca, 附带 PCAScikitLearn 的 numpy 实现)
# ======================================================================

def pca(points):
    """返回 (pc1, v3, explained_variance_ratio)。
    v3 是 C# pca() 返回的"重建向量"；建议闭环直接用 pc1。"""
    X = np.asarray(points, dtype=float)
    n = len(X)
    Xc = X - X.mean(axis=0)
    cov = Xc.T @ Xc / (n - 1)
    evals, evecs = np.linalg.eigh(cov)                     # 特征向量为列
    order = np.argsort(evals)[::-1]
    pc1 = evecs[:, order[0]]
    explained = evals[order] / evals.sum()

    # C# 里的 v3 重建(带 Acos/Atan2 的别扭写法, 等价于从球坐标重建)
    fj = math.atan2(pc1[1], pc1[0])
    d2 = math.acos(pc1[1])
    v3 = np.array([math.sin(d2) * math.cos(fj),
                   math.cos(d2),
                   math.sin(d2) * math.sin(fj)])
    return pc1, v3, explained


# ======================================================================
# 5. 焦点提取与概率拟合 (nsjy.cs)
# ======================================================================

def fit_focus(points, distances) -> np.ndarray:
    """nsjy.FitFocus: 球面交会线性化最小二乘 |p-F|² = d²"""
    n = len(points)
    p0 = np.asarray(points[0], float)
    d0 = distances[0]
    A = np.zeros((n - 1, 3))
    b = np.zeros(n - 1)
    for i in range(1, n):
        diff = np.asarray(points[i], float) - p0
        A[i - 1] = 2.0 * diff
        b[i - 1] = sqr_magnitude(points[i]) - sqr_magnitude(p0) \
                   - (distances[i] ** 2 - d0 ** 2)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    return sol


def extract_foci(points, prob_focus1, sigma1, prob_focus2, sigma2):
    """nsjy.ExtractFoci: 概率 → 距离 → 焦点"""
    n = len(points)
    if n < 4:
        raise ValueError("点数不足4")
    dist1 = [sigma1 * math.sqrt(-2.0 * math.log(clamp(p, 1e-6, 1.0 - 1e-6)))
             for p in prob_focus1]
    dist2 = [sigma2 * math.sqrt(-2.0 * math.log(clamp(p, 1e-6, 1.0 - 1e-6)))
             for p in prob_focus2]
    return fit_focus(points, dist1), fit_focus(points, dist2)


def fit_foci_by_probability(points, true_prob, init_f1, init_f2, a,
                            max_iter: int = 300, learning_rate: float = 0.001):
    """nsjy.FitFociByProbability: 梯度下降。
    注: C# 中 coef 缺因子 2(∂/∂x(f-t)² 的 2), 此处忠实保留。"""
    F1 = np.asarray(init_f1, float).copy()
    F2 = np.asarray(init_f2, float).copy()
    for _ in range(max_iter):
        g1 = np.zeros(3)
        g2 = np.zeros(3)
        loss = 0.0
        for i, p in enumerate(points):
            d1 = dist(p, F1)
            d2 = dist(p, F2)
            delta = d1 + d2 - 2.0 * a
            fit_prob = math.exp(-abs(delta) / (2.0 * a))
            diff = fit_prob - true_prob[i]
            loss += diff * diff
            sign = 1.0 if delta >= 0 else -1.0
            coef = diff * fit_prob * sign / (2.0 * a)
            if d1 > 1e-6:
                g1 += coef * (p - F1) / d1
            if d2 > 1e-6:
                g2 += coef * (p - F2) / d2
        F1 -= learning_rate * g1
        F2 -= learning_rate * g2
        if loss < 1e-12:
            break
    return F1, F2


def batch_probability(points, F1, F2, a) -> np.ndarray:
    """nsjy.BatchProbability。
    注: C# 原码为 -abs(delta)/2*a (优先级 bug, 指数被放大 a 倍),
    此处按数学意图修正为 -abs(delta)/(2*a)。"""
    fff = np.zeros(len(points))
    for i, p in enumerate(points):
        d1 = dist(p, F1)
        d2 = dist(p, F2)
        delta = d1 + d2 - 2.0 * a
        fff[i] = math.exp(-abs(delta) / (2.0 * a))
    return fff


# ======================================================================
# 6. DeviationCalculator (nsjy.cs 841-924 行)
# ======================================================================

def generate_lattice_1d(tau: complex, rng: int = 20) -> np.ndarray:
    """格 L = {m + n·tau}"""
    m = np.arange(-rng, rng + 1)
    n = np.arange(-rng, rng + 1)
    MM, NN = np.meshgrid(m, n)
    return (MM + NN * tau).ravel()


def nearest_distance(z: complex, lattice: np.ndarray) -> float:
    """最近格点距离 (numpy 向量化)"""
    return float(np.abs(z - lattice).min())


def compute_rms(values) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def compute_probabilities_from_taus(r30, r45, t1: complex, t2: complex, a: float):
    """返回 (probTotal, prob1, prob2, sigm), sigm = [2a, σ1, σ2]"""
    lat1 = generate_lattice_1d(t1, 20)
    lat2 = generate_lattice_1d(t2, 20)
    n = len(r30)
    d1 = np.array([nearest_distance(z, lat1) for z in r30])
    d2 = np.array([nearest_distance(z, lat2) for z in r45])
    sigma1 = compute_rms(d1)
    sigma2 = compute_rms(d2)
    prob1 = np.exp(-(d1 ** 2) / (2.0 * sigma1 * sigma1))
    prob2 = np.exp(-(d2 ** 2) / (2.0 * sigma2 * sigma2))
    sigm = np.array([2.0 * a, sigma1, sigma2])
    dist_total = d1 + d2 - sigm[0]
    prob_total = np.exp(-np.abs(dist_total) / sigm[0])
    return prob_total, prob1, prob2, sigm


# ======================================================================
# 7. LatticeAnalysis (nsjy.cs 380-508 行)
# ======================================================================

def estimate_lattice_basis(points) -> np.ndarray:
    """从复数对列表估计格基, 返回 4x4 矩阵(列 = 基向量)"""
    n = len(points)
    if n < 4:
        raise ValueError("至少需要 4 个点来估计格基")
    vectors = [np.array([p[0].real, p[0].imag, p[1].real, p[1].imag]) for p in points]
    basis = []
    for v in vectors:
        if len(basis) == 4:
            break
        if not basis:
            basis.append(v)
            continue
        M = np.column_stack(basis + [v])
        if np.linalg.matrix_rank(M) > len(basis):
            basis.append(v)
    if len(basis) < 4:
        raise ValueError("无法找到 4 个线性独立的向量，可能数据点不够或存在退化")
    return np.column_stack(basis)


def extract_period_matrix(B: np.ndarray) -> np.ndarray:
    """从 4x4 格基提取 2x2 周期矩阵 Ω (尝试所有 2 列组合)"""
    if B.shape != (4, 4):
        raise ValueError("B 必须是 4x4 矩阵")
    C = np.zeros((2, 4), dtype=complex)
    for j in range(4):
        C[0, j] = complex(B[0, j], B[1, j])
        C[1, j] = complex(B[2, j], B[3, j])
    for cols in itertools.combinations(range(4), 2):
        remaining = [c for c in range(4) if c not in cols]
        A = C[:, list(cols)]
        det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
        if abs(det) < 1e-12:
            continue
        Omega = np.linalg.inv(A) @ C[:, remaining]
        return Omega
    raise ValueError("无法从格基中提取周期矩阵：所有 2x2 子矩阵均奇异")


# ======================================================================
# 8. LLL 格基归约 (nsjy.cs 511-627 行)
# ======================================================================

def lll_reduce(basis: np.ndarray, delta: float = 0.75) -> np.ndarray:
    """LLL 归约。注: C# 原版用 static bstar 列表(有重复累积 bug),
    此处为每次调用重新计算的标准实现。"""
    n, m = basis.shape
    if m > n:
        raise ValueError("基向量个数不能超过维度")
    B = [np.array(basis[:, j], dtype=float) for j in range(m)]

    def recompute(start: int):
        """从 start 开始重算 Gram-Schmidt (原地更新 mu/Bn/Bstar)"""
        nonlocal Bstar, mu, Bn
        for i in range(start, m):
            v = B[i].copy()
            for j in range(i):
                if Bn[j] > 1e-20:
                    mu[i, j] = np.dot(B[i], Bstar[j]) / Bn[j]
                    v -= mu[i, j] * Bstar[j]
            Bstar[i] = v
            Bn[i] = max(np.dot(v, v), 0.0)

    Bstar = [None] * m
    mu = np.zeros((m, m))
    Bn = np.zeros(m)
    recompute(0)

    k = 1
    while k < m:
        # 尺寸约简
        for j in range(k - 1, -1, -1):
            if abs(mu[k, j]) > 0.5:
                q = round(mu[k, j])
                B[k] = B[k] - q * B[j]
                recompute(k)
        # Lovász 条件
        if Bn[k] >= (delta - mu[k, k - 1] * mu[k, k - 1]) * Bn[k - 1]:
            k += 1
        else:
            B[k], B[k - 1] = B[k - 1], B[k]
            recompute(max(0, k - 2))
            k = max(1, k - 1)
    return np.column_stack(B)


# ======================================================================
# 9. ComplexLinearFit (nsjy.cs 630-750 行)
# ======================================================================

def complex_linear_fit(z, w):
    """拟合 z ≈ a·w + b, 返回 (a, b, maxError)"""
    N = len(z)
    A = np.zeros((2 * N, 4))
    y = np.zeros(2 * N)
    for i in range(N):
        wR, wI = w[i].real, w[i].imag
        zR, zI = z[i].real, z[i].imag
        A[2 * i] = [wR, -wI, 1.0, 0.0]
        y[2 * i] = zR
        A[2 * i + 1] = [wI, wR, 0.0, 1.0]
        y[2 * i + 1] = zI
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    a = complex(sol[0], sol[1])
    b = complex(sol[2], sol[3])
    max_error = max(abs(a * w[i] + b - z[i]) for i in range(N))
    return a, b, max_error


# ======================================================================
# 10. LatticeRansac (nsjy.cs 752-839 行)
# ======================================================================

def find_largest_lattice(points, threshold: float = 1e-5, iterations: int = 1000,
                         seed: int | None = None):
    """RANSAC 找最大格内点子集, 返回 (inlierIndices, bestBasis)"""
    n = len(points)
    if n < 4:
        raise ValueError("至少需要 4 个点")
    vectors = [np.array([p[0].real, p[0].imag, p[1].real, p[1].imag]) for p in points]
    rng = np.random.default_rng(seed)
    best_inliers = []
    best_basis = None
    for _ in range(iterations):
        idx = rng.choice(n, 4, replace=False)
        B = np.column_stack([vectors[i] for i in idx])
        if np.linalg.matrix_rank(B) < 4:
            continue
        try:
            invB = np.linalg.inv(B)
        except np.linalg.LinAlgError:
            continue
        inliers = []
        for i in range(n):
            c_real = invB @ vectors[i]
            c_int = np.round(c_real)
            recon = B @ c_int
            if np.linalg.norm(recon - vectors[i]) < threshold:
                inliers.append(i)
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_basis = B
    if len(best_inliers) >= 4:
        bl = []
        for v in [vectors[i] for i in best_inliers]:
            if len(bl) == 4:
                break
            M = v.reshape(4, 1) if not bl else np.column_stack(bl + [v])
            if np.linalg.matrix_rank(M) > len(bl):
                bl.append(v)
        if len(bl) == 4:
            best_basis = np.column_stack(bl)
    return best_inliers, best_basis


# ======================================================================
# 11. n2sjy.cs 闭环: RefineModuliByAxis + EvaluateResiduals + NormalizeTau + Solve4
# ======================================================================

def normalize_tau(x: np.ndarray):
    """τ 归一化到基本域: Im>0、|Re|≤0.5、|τ|≥1 (n2sjy.NormalizeTau)"""
    for k in (0, 2):
        re, im = x[k], x[k + 1]
        if im <= 1e-4:
            im = 1e-4
        for _ in range(100):
            re -= round(re)
            mod2 = re * re + im * im
            if mod2 >= 1.0 - 1e-14:
                break
            re, im = -re / mod2, im / mod2
        x[k], x[k + 1] = re, im


def solve4(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """4x4 高斯消元(部分主元), 与 n2sjy.Solve4 一致"""
    aug = np.column_stack([A, b]).astype(float)
    n = 4
    for i in range(n):
        piv = i + int(np.argmax(np.abs(aug[i:, i])))
        aug[[i, piv]] = aug[[piv, i]]
        if abs(aug[i, i]) < 1e-15:
            continue
        aug[i + 1:] -= (aug[i + 1:, i:i + 1] / aug[i, i]) * aug[i:i + 1]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if abs(aug[i, i]) > 1e-15:
            x[i] = (aug[i, n] - aug[i, i + 1:n] @ x[i + 1:]) / aug[i, i]
        else:
            x[i] = aug[i, n]
    return x


def evaluate_residuals(x, pts, r30, r45, a, axis, w_dir, w_self, w_theory, t10, t20):
    """n2sjy.EvaluateResiduals:
    前 n 项=概率自洽(格概率 vs 焦点概率, ÷√n),
    后 3 项=方向闭合((F1-F2)∥axis),
    最后 4 项=理论正则(弱)。"""
    n = len(pts)
    t1 = complex(x[0], x[1])
    t2 = complex(x[2], x[3])
    prob_total, prob1, prob2, sigm = compute_probabilities_from_taus(r30, r45, t1, t2, a)
    F1, F2 = extract_foci(pts, prob1, sigm[1], prob2, sigm[2])

    r = np.zeros(n + 7)
    inv_sqrt_n = 1.0 / math.sqrt(n)
    for i in range(n):
        p = pts[i]
        d1 = dist(p, F1)
        d2 = dist(p, F2)
        delta = d1 + d2 - 2.0 * a
        prob_foci = math.exp(-abs(delta) / (2.0 * a))
        r[i] = w_self * (prob_total[i] - prob_foci) * inv_sqrt_n

    d = F1 - F2
    if sqr_magnitude(d) < 1e-12:
        d = axis.copy()
    d = d / np.linalg.norm(d)
    if np.dot(d, axis) < 0:
        d = -d
    e = d - axis
    r[n:n + 3] = w_dir * e
    r[n + 3] = w_theory * (x[0] - t10.real)
    r[n + 4] = w_theory * (x[1] - t10.imag)
    r[n + 5] = w_theory * (x[2] - t20.real)
    r[n + 6] = w_theory * (x[3] - t20.imag)
    return r, F1, F2


def _lm_loop(x, evaluate, axis, t10, t20, max_iter, angle_tol_deg, fd_h, label="",
             verbose=True):
    """LM 迭代主体。返回 (x, converged, last_angle_deg)。
    注意: 初始 x 不再调用 NormalizeTau(第一次用原始初值);
    仅对雅可比扰动 xp 与线搜索候选 xtry 做归一化(与 n2sjy.cs 一致)。"""
    r, F1, F2 = None, np.zeros(3), np.zeros(3)
    cost = float("inf")
    lam = 1e-3
    angle_deg = 180.0
    prefix = (label + " ") if label else ""

    for it in range(max_iter):
        r, F1, F2 = evaluate(x)
        cost = float(np.dot(r, r))

        d = F1 - F2
        angle_deg = 180.0
        if sqr_magnitude(d) > 1e-12:
            d = d / np.linalg.norm(d)
            if np.dot(d, axis) < 0:
                d = -d
            angle_deg = math.degrees(math.acos(clamp(np.dot(d, axis), -1.0, 1.0)))

        # n2sjy 的特殊重启: 前几步完全没动过且角度很大 → 试 (i,i)
        # (注: C# 里 angleDeg2 误用旧 dir, 这里用 dir0 修正)
        if it == 3 and angle_deg > 30.0 and x[0] == t10.real and x[1] == t10.imag:
            xt = np.array([0.0, 1.0, 0.0, 1.0])
            _, F11, F22 = evaluate(xt)
            d0 = F11 - F22
            angle2 = 180.0
            if sqr_magnitude(d0) > 1e-12:
                d0 = d0 / np.linalg.norm(d0)
                if np.dot(d0, axis) < 0:
                    d0 = -d0
                angle2 = math.degrees(math.acos(clamp(np.dot(d0, axis), -1.0, 1.0)))
            if angle2 < angle_deg:
                x = xt.copy()

        if verbose:
            print(f"[{prefix}iter {it}] cost={cost:.3e} angle={angle_deg:.3f}°  "
                  f"t1=({x[0]:.4f},{x[1]:.4f}) t2=({x[2]:.4f},{x[3]:.4f})")
        if angle_deg < angle_tol_deg:
            if verbose:
                print("闭环收敛")
            return x, True, angle_deg

        # 数值雅可比 J(m×4)
        J = np.zeros((len(r), 4))
        for k in range(4):
            xp = x.copy()
            xp[k] += fd_h
            normalize_tau(xp)
            rp2, _, _ = evaluate(xp)
            J[:, k] = (rp2 - r) / fd_h

        # LM 法方程 (JᵀJ + λ·diag)Δ = −Jᵀr
        A = J.T @ J
        g = J.T @ r
        Aaug = A.copy()
        for c in range(4):
            Aaug[c, c] += lam * (A[c, c] + 1e-12)
        delta = solve4(Aaug, g)

        xtry = x - delta
        normalize_tau(xtry)
        r2, _, _ = evaluate(xtry)
        cost2 = float(np.dot(r2, r2))
        if cost2 < cost:
            x = xtry
            lam = max(lam / 3.0, 1e-8)
        else:
            lam = min(lam * 3.0, 1e4)

    return x, angle_deg < angle_tol_deg, angle_deg


def refine_moduli_by_axis(pts, t10: complex, t20: complex, pca_axis,
                          max_iter: int = 100,
                          angle_tol_deg: float = 0.5,
                          fd_h: float = 1e-3,
                          w_dir: float = 20.0,
                          w_self: float = 1.0,
                          w_theory: float = 1e-3,
                          verbose=True):
    """n2sjy.RefineModuliByAxis: LM 闭环优化 τ, 使 (F1-F2) 平行于 PCA 主轴。
    第 1 次直接用原始初值(不调用 NormalizeTau); 若未收敛, 依次用
    标准模 (0,1),(0,1) 与 理论归一化 起点重试, 事后取角度更小者 (best-of-N)。"""
    n = len(pts)
    if n < 4:
        raise ValueError("至少 4 个点")

    rp = compute_rp(pts)
    cone = math.asin(math.cos(math.radians(30.0)) / math.pi)
    a = rp * (1.0 + math.sin(cone))
    r30 = [inverse_th4(p, 30, rp) for p in pts]
    r45 = [inverse_th4(p, 45, rp) for p in pts]
    axis = np.asarray(pca_axis, float)
    axis = axis / np.linalg.norm(axis)

    def evaluate(xv):
        return evaluate_residuals(xv, pts, r30, r45, a, axis,
                                  w_dir, w_self, w_theory, t10, t20)

    # 第 1 次尝试: 原始初值 t1,t2 直接输入, 不调用 NormalizeTau
    x0 = np.array([t10.real, t10.imag, t20.real, t20.imag], dtype=float)
    x, converged, angle_deg = _lm_loop(x0.copy(), evaluate, axis, t10, t20,
                                       max_iter, angle_tol_deg, fd_h,
                                       verbose=verbose)

    # 追加判断: 原始初值未收敛 → 候选起点重试, 事后取角度更小者 (best-of-N)
    if not converged:
        xnorm = x0.copy()
        normalize_tau(xnorm)
        starts = [np.array([0.0, 1.0, 0.0, 1.0])]      # 标准模 (0,1),(0,1)
        if not np.allclose(xnorm, starts[0], atol=1e-9):
            starts.append(xnorm)                        # 理论归一化 (兼容依赖它的案例)
        for k, xn in enumerate(starts):
            label = "标准模" if k == 0 else "归一化理论"
            x2, conv2, angle2 = _lm_loop(xn, evaluate, axis, t10, t20,
                                         max_iter, angle_tol_deg, fd_h,
                                         label=label, verbose=verbose)
            if angle2 < angle_deg:
                if verbose:
                    print(f"{label}重试 angle={angle2:.3f}° < 当前 {angle_deg:.3f}°, 采用")
                x, converged, angle_deg = x2, conv2, angle2
                if converged:
                    break
            elif verbose:
                print(f"{label}重试 angle={angle2:.3f}° ≥ 当前 {angle_deg:.3f}°, 保留当前")

    r, F1, F2 = evaluate(x)
    return complex(x[0], x[1]), complex(x[2], x[3]), F1, F2


# ======================================================================
# 12. 主流程 (对应 nsjy.Start + n2sjy.Start)
# ======================================================================

def load_points(path: str):
    """加载点集: JSON 数组 [[x,y,z],...] 或每行 'x y z'"""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            for key in ("points", "Vector3List", "data"):
                if key in data and isinstance(data[key], list):
                    data = data[key]
                    break
        return [np.array(p, dtype=float) for p in data]
    except json.JSONDecodeError:
        pts = []
        for line in text.splitlines():
            parts = line.split()
            if len(parts) >= 3:
                pts.append(np.array([float(parts[0]), float(parts[1]), float(parts[2])]))
        return pts


def random_points(n: int = 400, seed: int = 0) -> list:
    """单位球内均匀随机点 (等价 UnityEngine.Random.insideUnitSphere)"""
    rng = np.random.default_rng(seed)
    dirs = rng.standard_normal((n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    radii = rng.random(n) ** (1.0 / 3.0)
    return [d * r for d, r in zip(dirs, radii)]


def main():
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        points = load_points(sys.argv[1])
    else:
        points = random_points(400, seed=0)

    # ---- 预处理 (nsjy.Start) ----
    rp = compute_rp(points)
    cone = math.asin(math.cos(math.radians(30.0)) / math.pi)
    a = rp * (1.0 + math.sin(cone))
    print(f"n={len(points)}  rp={rp:.6f}  a={a:.6f}")

    r30 = [inverse_th4(p, 30, rp) for p in points]
    r45 = [inverse_th4(p, 45, rp) for p in points]
    pairs, r45b, r30b, r74b = yzqx(points, rp)

    # ---- 理论模量 (nsjy.Start) ----
    Kk = elliptic_integral_k(0.5 * 0.5)
    K_sqrt3_2 = elliptic_integral_k((math.sqrt(3) / 2.0) ** 2)
    K_2 = complex(0.5 * Kk.real, -0.5 * K_sqrt3_2.real)
    gamma14 = gamma(0.25)
    K_inv_sqrt2 = gamma14 * gamma14 / (4.0 * math.sqrt(math.pi))
    K_sqrt2 = complex(1.0 / math.sqrt(2), -1.0 / math.sqrt(2)) * K_inv_sqrt2
    t10, t20 = compute_taus()

    K_sin16 = Carlsonfk.K(1.0 / math.sin(math.radians(16.0)))
    K_sin74 = Carlsonfk.K(1.0 / math.sin(math.radians(74.0)))
    aot = K_sqrt2 - K_2 - K_sin16
    aot2 = K_sin74 - K_2 - K_sqrt2
    KF2 = Carlsonfk.NK(aot2)
    KF = Carlsonfk.NK(aot)
    print(f"aot={aot} aot2={aot2}")
    print(f"KF={KF}  KF2={KF2}")

    # ---- PCA 轴 ----
    pc1, v3, explained = pca(points)
    print(f"explained={explained}  pc1={pc1}")
    axis = normalized(v3)          # 与 n2sjy.Start 一致; 建议改用 pc1
    # axis = normalized(pc1)

    # ---- 概率 → 焦点 ----
    probs = compute_probabilities_from_taus(r30, r45, t10, t20, a)
    F1, F2 = extract_foci(points, probs[1], probs[3][1], probs[2], probs[3][2])
    print(f"ExtractFoci: F1={F1} F2={F2}")

    f1, f2 = fit_foci_by_probability(points, probs[0], F1, F2, a)
    fff = batch_probability(points, f1, f2, a)
    print("fff =", np.array2string(fff, precision=6, max_line_width=120))
    print("rms(fff - probTotal) =",
          float(np.sqrt(np.mean((fff - probs[0]) ** 2))))

    # ---- 闭环 (n2sjy.Start) ----
    t1f, t2f, F1f, F2f = refine_moduli_by_axis(points, t10, t20, axis)
    d = F1f - F2f
    print(f"闭环结果: t1f={t1f} t2f={t2f}")
    print(f"F1f={F1f} F2f={F2f}  dir={normalized(d)}  axis={axis}")

    # ---- 格分析演示 (--lattice) ----
    if "--lattice" in sys.argv:
        print("\n--- LatticeAnalysis ---")
        B = estimate_lattice_basis(pairs)
        print("basis =\n", B)
        print("rank =", np.linalg.matrix_rank(B))
        Omega = extract_period_matrix(B)
        print("period matrix Ω =\n", Omega)
        print("\n--- LLL ---")
        Bred = lll_reduce(B, 0.99)
        print("reduced basis =\n", Bred)
        print("lengths:", [np.linalg.norm(Bred[:, j]) for j in range(4)])
        print("\n--- LatticeRansac ---")
        inliers, basis = find_largest_lattice(pairs, threshold=1e-3, iterations=200, seed=0)
        print(f"inliers: {len(inliers)}/{len(pairs)}")


if __name__ == "__main__":
    main()
