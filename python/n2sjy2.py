import math
import numpy as np
from scipy.optimize import least_squares
from mpmath import mp, mpc, ellipk, sqrt as msqrt
from sklearn.decomposition import PCA
import random

# ========== 复数椭圆积分 K(k) ==========
def K_complex(k):
    """完全椭圆积分 K(k)，使用 mpmath，参数为 m = k²"""
    k = mpc(k)
    return complex(ellipk(k**2))

# ========== ComputeTaus ==========
def compute_taus():
    K2 = K_complex(2.0)
    Kprime2 = K_complex(msqrt(mpc(1 - 4)))  # K(i√3)
    tau1 = complex(1j) * Kprime2 / K2

    Ksqrt2 = K_complex(mpc(msqrt(mpc(2))))
    KprimeSqrt2 = K_complex(msqrt(mpc(1 - 2)))  # K(i)
    tau2 = complex(1j) * KprimeSqrt2 / Ksqrt2

    if tau1.imag < 0:
        tau1 = -tau1
    if tau2.imag < 0:
        tau2 = -tau2
    return tau1, tau2

# ========== 几何工具 ==========
def compute_rp(points):
    """平均到原点的距离"""
    return np.mean([np.linalg.norm(p) for p in points]) if len(points) > 0 else 0.0

def pca_axis(points):
    """返回第一主成分单位向量"""
    data = np.array(points)
    pca = PCA(n_components=3)
    pca.fit(data)
    pc1 = pca.components_[0]
    return pc1 / np.linalg.norm(pc1)

# ========== 逆映射 InverseTh4 ==========
def inverse_th4(vs3, d2_deg, rp):
    d2 = math.radians(d2_deg)
    x, y, z = vs3
    rs = math.sqrt(x*x + y*y)
    zs = z
    phis = math.atan2(y, x)
    if phis < 0:
        phis += 2 * math.pi

    r = math.sqrt(rs*rs + zs*zs)
    sin_d2 = math.sin(d2)
    # 注意：原代码中 thetas 计算用 phis / sin_d2，这里保持一致
    thetas = phis / sin_d2
    thetas = thetas % (2 * math.pi)

    Rs = (r / rp) ** (1.0 / sin_d2)
    u = Rs * math.cos(thetas)
    v = Rs * math.sin(thetas)
    return complex(u, v)

# ========== yzqx ==========
def yzqx(points, rp):
    count = len(points)
    r45 = [0]*count
    r30 = [0]*count
    r74 = [0]*count
    for i, p in enumerate(points):
        r45[i] = inverse_th4(p, 45, rp)
        r30[i] = inverse_th4(p, 30, rp)
        r74[i] = inverse_th4(p, 74, rp)

    raw_list = [(r30[i], r45[i]) for i in range(count)]
    valid_list = []
    for x, y in raw_list:
        if (math.isnan(x.real) or math.isnan(x.imag) or math.isinf(x.real) or math.isinf(x.imag) or
            math.isnan(y.real) or math.isnan(y.imag) or math.isinf(y.real) or math.isinf(y.imag)):
            continue
        valid_list.append((x, y))

    if not valid_list:
        return [], r45, r30, r74

    threshold = 1e10
    filtered = [t for t in valid_list if abs(t[0]) < threshold and abs(t[1]) < threshold]
    if not filtered:
        return [], r45, r30, r74

    max_mag = max([abs(t[0]) for t in filtered] + [abs(t[1]) for t in filtered])
    if max_mag > 0:
        normalized = [(t[0]/max_mag, t[1]/max_mag) for t in filtered]
        return normalized, r45, r30, r74
    else:
        return filtered, r45, r30, r74

# ========== 焦点拟合相关 ==========
def fit_foci_by_probability(points, true_prob, init_f1, init_f2, a,
                            max_iter=300, learning_rate=0.0001, lambda_sep=1.0):
    F1 = np.array(init_f1, dtype=float)
    F2 = np.array(init_f2, dtype=float)
    eps = 1e-3

    for _ in range(max_iter):
        grad_f1 = np.zeros(3)
        grad_f2 = np.zeros(3)
        loss = 0.0

        for i, p in enumerate(points):
            p = np.array(p)
            d1 = np.linalg.norm(p - F1)
            d2 = np.linalg.norm(p - F2)
            safe_d1 = max(d1, eps)
            safe_d2 = max(d2, eps)

            delta = d1 + d2 - 2.0 * a
            fit_prob = math.exp(-abs(delta) / (2.0 * a))
            diff = fit_prob - true_prob[i]
            loss += diff * diff

            sign = 1.0 if delta >= 0 else -1.0
            coef = diff * fit_prob * sign / (2.0 * a)

            grad1 = coef * (p - F1) / safe_d1
            grad2 = coef * (p - F2) / safe_d2

            # 单点梯度裁剪
            norm1 = np.linalg.norm(grad1)
            if norm1 > 1.0:
                grad1 *= 1.0 / norm1
            norm2 = np.linalg.norm(grad2)
            if norm2 > 1.0:
                grad2 *= 1.0 / norm2

            grad_f1 += grad1
            grad_f2 += grad2

        # 分离正则
        sep = F1 - F2
        sep_loss = lambda_sep * np.dot(sep, sep)
        loss += sep_loss
        grad_f1 += 2.0 * lambda_sep * sep
        grad_f2 -= 2.0 * lambda_sep * sep

        # 整体梯度裁剪
        norm_f1 = np.linalg.norm(grad_f1)
        if norm_f1 > 5.0:
            grad_f1 *= 5.0 / norm_f1
        norm_f2 = np.linalg.norm(grad_f2)
        if norm_f2 > 5.0:
            grad_f2 *= 5.0 / norm_f2

        F1 -= learning_rate * grad_f1
        F2 -= learning_rate * grad_f2

        if not (np.all(np.isfinite(F1)) and np.all(np.isfinite(F2))):
            print("数值发散，停止迭代")
            break

        if loss < 1e-12:
            break

    return F1, F2

def batch_probability(F1, F2, points, a):
    probs = []
    for p in points:
        p = np.array(p)
        d1 = np.linalg.norm(p - F1)
        d2 = np.linalg.norm(p - F2)
        delta = d1 + d2 - 2.0 * a
        probs.append(math.exp(-abs(delta) / (2.0 * a)))
    return probs

def extract_foci(points, prob_focus1, sigma1, prob_focus2, sigma2):
    n = len(points)
    if n < 4 or len(prob_focus1) != n or len(prob_focus2) != n:
        raise ValueError("点数和概率数组长度不匹配或点数不足4")

    dist1 = []
    dist2 = []
    for i in range(n):
        p1 = min(max(prob_focus1[i], 1e-6), 1-1e-6)
        p2 = min(max(prob_focus2[i], 1e-6), 1-1e-6)
        dist1.append(sigma1 * math.sqrt(-2.0 * math.log(p1)))
        dist2.append(sigma2 * math.sqrt(-2.0 * math.log(p2)))

    F1 = fit_focus(points, dist1)
    F2 = fit_focus(points, dist2)
    return F1, F2

def fit_focus(points, distances):
    """最小二乘拟合球心"""
    n = len(points)
    p0 = np.array(points[0])
    d0 = distances[0]

    A = np.zeros((n-1, 3))
    b = np.zeros(n-1)
    for i in range(1, n):
        diff = np.array(points[i]) - p0
        A[i-1, :] = 2.0 * diff
        b[i-1] = (np.dot(points[i], points[i]) - np.dot(p0, p0)
                  - (distances[i]**2 - d0**2))

    # 最小二乘解
    solution, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return solution

# ========== DeviationCalculator.ComputeProbabilitiesFromTaus ==========
def generate_lattice_1d(tau, rng=20):
    lattice = []
    for m in range(-rng, rng+1):
        for n in range(-rng, rng+1):
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

def compute_probabilities_from_taus(r30, r45, t1, t2, a):
    lattice1 = generate_lattice_1d(t1)
    lattice2 = generate_lattice_1d(t2)

    n = len(r30)
    d1_arr = np.zeros(n)
    d2_arr = np.zeros(n)
    for i in range(n):
        d1_arr[i] = nearest_distance(r30[i], lattice1)
        d2_arr[i] = nearest_distance(r45[i], lattice2)

    sigma1 = math.sqrt(np.mean(d1_arr**2))
    sigma2 = math.sqrt(np.mean(d2_arr**2))
    sigmas = np.array([2*a, sigma1, sigma2])

    prob1 = np.exp(-(d1_arr**2) / (2 * sigma1**2))
    prob2 = np.exp(-(d2_arr**2) / (2 * sigma2**2))

    dist_total = d1_arr + d2_arr - 2*a
    prob_total = np.exp(-np.abs(dist_total) / (2*a))

    return prob_total, prob1, prob2, sigmas

# ========== LM 优化相关 ==========
def normalize_tau(x,perturb_if_unit_mod=True, perturb_scale=0.01, tolerance=1e-3):
    """τ 归一化：模群基本域 + 硬约束"""
    for k in range(0, 4, 2):
        re = x[k]
        im = x[k+1]

        # 基本域归一化
        for _ in range(100):
            re -= round(re)
            mod2 = re*re + im*im
            if mod2 >= 1.0 - 1e-14:
                break
            den = mod2
            new_re = -re / den
            new_im = im / den
            re, im = new_re, new_im

        # 硬约束
       
        x[k] = re
        x[k+1] = im

def solve4(A, b):
    """求解 4x4 线性方程组"""
    return np.linalg.solve(A, b)

def evaluate_residuals(x, pts, axis, r30, r45, a,
                       w_dir=20.0, w_self=1.0, w_theory=1e-3, w_i=1.0,
                       t10=None, t20=None):
    n = len(pts)
    t1 = complex(x[0], x[1])
    t2 = complex(x[2], x[3])

    # 概率计算
    prob_total, prob1, prob2, sigmas = compute_probabilities_from_taus(r30, r45, t1, t2, a)

    # 提取焦点
    F1, F2 = extract_foci(pts, prob1, sigmas[1], prob2, sigmas[2])

    r = np.zeros(n + 9)
    inv_sqrt_n = 1.0 / math.sqrt(n)

    # 概率自洽
    for i in range(n):
        p = np.array(pts[i])
        d1 = np.linalg.norm(p - F1)
        d2 = np.linalg.norm(p - F2)
        delta = d1 + d2 - 2.0 * a
        prob_foci = math.exp(-abs(delta) / (2.0 * a))
        r[i] = w_self * (prob_total[i] - prob_foci) * inv_sqrt_n

    # 方向闭合
    dir_vec = F1 - F2
    if np.linalg.norm(dir_vec) < 1e-12:
        dir_vec = axis
    dir_vec = dir_vec / np.linalg.norm(dir_vec)
    if np.dot(dir_vec, axis) < 0:
        dir_vec = -dir_vec
    e = dir_vec - axis
    r[n] = w_dir * e[0]
    r[n+1] = w_dir * e[1]
    r[n+2] = w_dir * e[2]

    # 理论正则
    if t10 is not None and t20 is not None:
        r[n+3] = w_theory * (x[0] - t10.real)
        r[n+4] = w_theory * (x[1] - t10.imag)
        r[n+5] = w_theory * (x[2] - t20.real)
        r[n+6] = w_theory * (x[3] - t20.imag)

    # 软约束拉向 (0,1)
    im1 = x[1]
    dev1 = 0
    for i in range(100):
        if im1 - 2**dev1 < 0:
            break
        dev1 += 1
    dy = 1.0
    for i in range(int(dev1)):
        dy += 1.0 / (2**i)

   
    r[n+7] = dy * (x[1] - 1)
   
    r[n+8] = dy * (x[3] - 1)

    return r, F1, F2

def refine_moduli_by_axis(pts, r30, r45, a, t10, t20, pca_axis,
                          max_iter=1, angle_tol_deg=0.5, fd_h=1e-3,
                          w_dir=20.0, w_self=1.0, w_theory=1e-3, w_i=1.0):
    n = len(pts)
    if n < 4:
        raise ValueError("至少 4 个点")

    axis = pca_axis / np.linalg.norm(pca_axis)
    x = np.array([t10.real, t10.imag, t20.real, t20.imag], dtype=float)
    r = None
    F1 = np.zeros(3)
    F2 = np.zeros(3)
    lam = 1e-3

    for it in range(max_iter):
        r, F1, F2 = evaluate_residuals(x, pts, axis, r30, r45, a,
                                        w_dir, w_self, w_theory, w_i, t10, t20)
        cost = np.sum(r**2)

        dir_vec = F1 - F2
        angle_deg = 180.0
        if np.linalg.norm(dir_vec) > 1e-12:
            dir_vec = dir_vec / np.linalg.norm(dir_vec)
            if np.dot(dir_vec, axis) < 0:
                dir_vec = -dir_vec
            cos_ang = np.clip(np.dot(dir_vec, axis), -1, 1)
            angle_deg = math.degrees(math.acos(cos_ang))

        print(f"[iter {it}] cost={cost:.3E} angle={angle_deg:.3f}° "
              f"t1=({x[0]:.4f},{x[1]:.4f}) t2=({x[2]:.4f},{x[3]:.4f})")

        if angle_deg < angle_tol_deg:
            print("闭环收敛")
            break

        # 数值雅可比
        J = np.zeros((len(r), 4))
        for k in range(4):
            xp = x.copy()
            xp[k] += fd_h
            normalize_tau(x, perturb_if_unit_mod=True, perturb_scale=0.02)
         
            rp2, _, _ = evaluate_residuals(xp, pts, axis, r30, r45, a,
                                            w_dir, w_self, w_theory, w_i, t10, t20)
            J[:, k] = (rp2 - r) / fd_h

        # LM 法方程
        A = J.T @ J
        g = J.T @ r
        A_aug = A.copy()
        for c in range(4):
            A_aug[c, c] += lam * (A[c, c] + 1e-12)
        delta = solve4(A_aug, g)

        # 线搜索
        xtry = x - delta
        normalize_tau(x, perturb_if_unit_mod=True, perturb_scale=0.02)
        r2, _, _ = evaluate_residuals(xtry, pts, axis, r30, r45, a,
                                      w_dir, w_self, w_theory, w_i, t10, t20)
        cost2 = np.sum(r2**2)
        if cost2 < cost:
            x = xtry
            lam = max(lam / 3.0, 1e-8)
        else:
            lam = min(lam * 3.0, 1e4)

    r, F1, F2 = evaluate_residuals(x, pts, axis, r30, r45, a,
                                    w_dir, w_self, w_theory, w_i, t10, t20)
    return complex(x[0], x[1]), complex(x[2], x[3]), F1, F2

# ========== 主流程示例 ==========

if __name__ == "__main__":
    np.random.seed(10)
    points = np.random.uniform(-1, 1, (200, 3))

    rp = compute_rp(points)
    d2 = math.asin(math.cos(math.radians(30)) / math.pi)
    a = rp * (1 + math.sin(d2))

    t1_theory, t2_theory = compute_taus()
    print("理论 tau:", t1_theory, t2_theory)

    normalized_list, r45, r30, r74 = yzqx(points, rp)

    prob_total, prob1, prob2, sigmas = compute_probabilities_from_taus(r30, r45, t1_theory, t2_theory, a)
    F1_init, F2_init = extract_foci(points, prob1, sigmas[1], prob2, sigmas[2])

    # 用概率拟合更精确焦点
    F1_fit, F2_fit = fit_foci_by_probability(points, prob_total, F1_init, F2_init, a)

    # 初始化当前焦点和初始方向
    f1, f2 = F1_fit, F2_fit
    init_dir = (f1 - f2) / np.linalg.norm(f1 - f2)   # 初始方向

    v_pca = pca_axis(points)

    # 初始模参数：建议使用理论值，而不是 i
    t1, t2 = complex(0,1), complex(0,1)  # 初始为 i
    max_outer_iter = 300   # 先减小外循环次数

    for i in range(max_outer_iter):
        prob_total_new, _, _, _ = compute_probabilities_from_taus(r30, r45, t1, t2, a)

        # 内层优化：使用当前 t1,t2 作为初值，迭代次数减小到 5~10
        t1_new, t2_new, F1_new, F2_new = refine_moduli_by_axis(
            points, r30, r45, a, complex(0,1), complex(0,1),
            (f1 - f2) / np.linalg.norm(f1 - f2),
            max_iter=50  # 先小，调试稳定后再增大
        )

        # 重新拟合焦点
        f1_new, f2_new = fit_foci_by_probability(points, prob_total_new, F1_new, F2_new, a)

        # 计算与初始方向的夹角
        dir_vec = (f1_new - f2_new) / np.linalg.norm(f1_new - f2_new)
        cos_angle1 = np.clip(np.dot(dir_vec, init_dir), -1.0, 1.0)
        angle1 = math.degrees(math.acos(cos_angle1))

        # 计算与 PCA 主轴的夹角（重要）
        cos_angle_pca = np.clip(np.dot(dir_vec, v_pca), -1.0, 1.0)
        angle_pca = math.degrees(math.acos(cos_angle_pca))

       

        # 更新变量
        f1, f2 = f1_new, f2_new
        t1, t2 = t1_new, t2_new

        print(f"iter {i}: angle1={angle1:.2f}°, angle_pca={angle_pca:.2f}°")
