import numpy as np
import math
import cmath
from scipy.special import gamma
import pandas as pd
from mpmath import ellipk as mp_ellipk  # 统一放在顶部导入

# ---------- 工具函数 ----------
def gamma_func(x):
    return gamma(x)

# ---------- 核心几何/概率函数（完全保持您的原有逻辑）----------
def inverse_th4(p, angle_deg, rp):
    angle = angle_deg * math.pi / 180
    x, y, z = p
    rs = math.sqrt(x*x + y*y)
    zs = z
    phis = math.atan2(y, x)
    if phis < 0:
        phis += 2*math.pi
    r_val = math.sqrt(rs*rs + zs*zs)
    sin_a = math.sin(angle)
    if abs(sin_a) < 1e-12:
        return complex(0,0)
    thetas = phis / sin_a
    while thetas < 0:
        thetas += 2*math.pi
    while thetas >= 2*math.pi:
        thetas -= 2*math.pi
    Rs = (r_val / rp) ** (1.0 / sin_a)
    u = Rs * math.cos(thetas)
    v = Rs * math.sin(thetas)
    return complex(u, v)

def yzqx(points, rp):
    r45 = np.array([inverse_th4(p, 45, rp) for p in points])
    r30 = np.array([inverse_th4(p, 30, rp) for p in points])
    r74 = np.array([inverse_th4(p, 74, rp) for p in points])
    all_mod = np.concatenate([np.abs(r45), np.abs(r30), np.abs(r74)])
    max_mod = np.max(all_mod)
    if max_mod > 0:
        r45 /= max_mod
        r30 /= max_mod
        r74 /= max_mod
    return r45, r30, r74

def compute_taus():
    K2 = complex(mp_ellipk(2.0))
    Kprime2 = complex(mp_ellipk(1j * math.sqrt(3)))
    tau1 = 1j * Kprime2 / K2
    if tau1.imag < 0: tau1 = -tau1
    Ksqrt2 = complex(mp_ellipk(math.sqrt(2)))
    KprimeSqrt2 = complex(mp_ellipk(1j))
    tau2 = 1j * KprimeSqrt2 / Ksqrt2
    if tau2.imag < 0: tau2 = -tau2
    return tau1, tau2

def generate_lattice_1d(tau, range_val=20):
    lattice = []
    for m in range(-range_val, range_val+1):
        for n in range(-range_val, range_val+1):
            lattice.append(m + n * tau)
    return lattice

def nearest_distance(z, lattice):
    min_sq = float('inf')
    for lam in lattice:
        diff = z - lam
        sq = diff.real**2 + diff.imag**2
        if sq < min_sq:
            min_sq = sq
    return math.sqrt(min_sq)

def compute_probabilities_from_taus(r30, r45, tau1, tau2, a):
    lattice1 = generate_lattice_1d(tau1)
    lattice2 = generate_lattice_1d(tau2)
    n = len(r30)
    d1_arr = np.zeros(n)
    d2_arr = np.zeros(n)
    for i in range(n):
        d1_arr[i] = nearest_distance(r30[i], lattice1)
        d2_arr[i] = nearest_distance(r45[i], lattice2)
    sigma1 = math.sqrt(np.mean(d1_arr**2))
    sigma2 = math.sqrt(np.mean(d2_arr**2))
    prob1 = np.exp(- (d1_arr**2) / (2 * sigma1**2))
    prob2 = np.exp(- (d2_arr**2) / (2 * sigma2**2))
    sigma0 = 2 * a
    dist_total = d1_arr + d2_arr - sigma0
    prob_total = np.exp(- np.abs(dist_total) / sigma0)
    return prob_total, prob1, prob2, np.array([sigma0, sigma1, sigma2])

def fit_focus(points, distances):
    n = len(points)
    p0 = points[0]
    d0 = distances[0]
    A = []
    b = []
    for i in range(1, n):
        pi = points[i]
        diff = pi - p0
        A.append([2*diff[0], 2*diff[1], 2*diff[2]])
        rhs = np.dot(pi, pi) - np.dot(p0, p0) - (distances[i]**2 - d0**2)
        b.append(rhs)
    A = np.array(A)
    b = np.array(b)
    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return sol

def extract_foci(points, prob1, sigma1, prob2, sigma2):
    prob1 = np.clip(prob1, 1e-12, 1-1e-12)
    prob2 = np.clip(prob2, 1e-12, 1-1e-12)
    dist1 = sigma1 * np.sqrt(-2 * np.log(prob1))
    dist2 = sigma2 * np.sqrt(-2 * np.log(prob2))
    F1 = fit_focus(points, dist1)
    F2 = fit_focus(points, dist2)
    return F1, F2

def pca(points):
    cov = np.cov(points.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    idx = np.argmax(eigvals)
    v = eigvecs[:, idx].real
    return v / np.linalg.norm(v)

def batch_probability(F1, F2, points, a):
    probs = np.zeros(len(points))
    for i, p in enumerate(points):
        d1 = np.linalg.norm(p - F1)
        d2 = np.linalg.norm(p - F2)
        delta = d1 + d2 - 2*a
        probs[i] = np.exp(-abs(delta) / (2*a))
    return probs

# ---------- 封装单次试验（生成一个点集，返回所需数据）----------
def run_single_trial(seed, n_points=200):
    # 1. 使用固定种子生成点集
    rng = np.random.default_rng(seed)
    u = rng.normal(0, 1, (n_points, 3))
    r = rng.random(n_points) ** (1/3)
    points = (u / np.linalg.norm(u, axis=1, keepdims=True)) * r[:, None]

    # 2. 基础参数
    rp = np.mean(np.linalg.norm(points, axis=1))
    d2 = math.asin(math.cos(30 * math.pi/180) / math.pi)
    a = rp * (1 + math.sin(d2))
    e3 = math.cos(d2) * 2 / (1 + math.sin(d2))
    c = rp * math.cos(d2) * e3

    # 3. 复数映射
    r45, r30, r74 = yzqx(points, rp)

    # 4. 计算 Taus（常数，可预先计算以提升性能，这里保持每次调用以保持独立）
    t1, t2 = compute_taus()
    t11 = complex(0,1)
    t22 = complex(0,1)

    # 5. 格点概率
    prob_total1, prob1, prob2, sigm = compute_probabilities_from_taus(r30, r45, t1, t2, a)
    prob_total2, prob1_2, prob2_2, sigm2 = compute_probabilities_from_taus(r30, r45, t11, t22, a)

    # 6. 提取焦点
    F1, F2 = extract_foci(points, prob1, sigm[1], prob2, sigm[2])
    F11, F22 = extract_foci(points, prob1_2, sigm2[1], prob2_2, sigm2[2])

    # 7. PCA
    v3pca = pca(points)

    # 8. 计算四组概率（对应 C# 中的 s1, s2, s3, s4）
    s1 = batch_probability(F1, F2, points, a)          # P1
    s12 = batch_probability(F11, F22, points, a)       # P2
    s3 = prob_total1                                   # P3
    s4 = prob_total2                                   # P4

    # 9. 计算实际概率（对应 C# 中的 vs / s13）
    f1 = c * v3pca
    f2 = -c * v3pca
    s13 = batch_probability(f1, f2, points, a)         # 实际概率

    # 10. 计算角度 angleDeg（对应 C# 中的 Mathf.Acos(Vector3.Dot(v3, v31))）
    v3 = (F1 - F2) / np.linalg.norm(F1 - F2)
    v31 = (F11 - F22) / np.linalg.norm(F11 - F22)
    dot_val = np.clip(np.dot(v3, v31), -1.0, 1.0)
    angle_deg = math.acos(dot_val) * 180 / math.pi

    # 返回：角度，四组概率的第一个值，实际概率的第一个值
    return angle_deg, s1[0], s12[0], s3[0], s4[0], s13[0]

# ---------- 主验证循环 ----------
if __name__ == "__main__":
    # ---------- 配置参数 ----------
    max_trials = 5000            # 可适当增加
    angle_low =90  # 角度下限（不包含）
    angle_high =164   # 角度上限（不包含）
    sim_threshold = 0.01         # 认为“相近”的阈值

    seed = 0
    valid_count = 0              # 在角度范围内的总样本数
    printed_count = 0            # 实际打印的样本数（经筛选后）
    max_print = 200              # 最多打印 200 行，避免刷屏

    print(f"开始采集 {angle_low}° < Angle < {angle_high}° 的样本，剔除 |p1-p2| < {sim_threshold} 或 |p3-p4| < {sim_threshold} 的样本")
    print("输出格式：Seed | Angle | p1, p2, p3, p4 | actual")
    print("=" * 80)

    while seed < max_trials:
        angle_deg, p1, p2, p3, p4, actual = run_single_trial(seed)
        seed += 1

        # ---------- 角度范围条件 ----------
        if angle_low < angle_deg < angle_high:
            valid_count += 1

            # ---------- 筛选条件：剔除相近值 ----------
            diff12 = abs(p1 - p2)
            diff34 = abs(p3 - p4)
            if diff12 < sim_threshold or diff34 < sim_threshold:
                # 跳过该样本，不打印
                continue

            # 如果已经打印够了，就不继续打印，但计数仍在继续
            if printed_count >= max_print:
                continue

            # 打印满足条件的样本
            print(f"Seed {seed-1:4d} | Angle={angle_deg:.3f}° | "
                  f"p1={p1:.6f}, p2={p2:.6f}, p3={p3:.6f}, p4={p4:.6f} | "
                  f"actual={actual:.6f}  (diff12={diff12:.4f}, diff34={diff34:.4f})")
            printed_count += 1

        # 进度提示
        if seed % 1000 == 0:
            print(f"进度：已尝试 {seed} 次，找到 {angle_low}°<Angle<{angle_high}° 的样本 {valid_count} 个，已打印筛选后 {printed_count} 个")

    print("\n" + "=" * 60)
    print(f"总共尝试 {seed} 次，找到 {angle_low}° < Angle < {angle_high}° 的样本数: {valid_count}")
    print(f"其中满足 |p1-p2|>={sim_threshold} 且 |p3-p4|>={sim_threshold} 的样本数: {printed_count} (仅打印前 {max_print} 个)")
