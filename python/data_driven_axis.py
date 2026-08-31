# -*- coding: utf-8 -*-
"""
数据驱动无 PCA 的方向不动点迭代 (fast, 无震荡) — 含内层/外层次数影响验证

设计:
  目标1 快速收敛: 内层 LM 热启动(沿用上一轮 t1,t2, 不再从 (i,i) 重启), 外环阻尼 ω
  目标2 数据驱动收敛: 判据 = 方向不再变化 |d_{k+1}-d_k| < tol (无 PCA)
  目标3 无 PCA: 内层方向目标 = 当前数据焦点方向 d (自指), 全程不出现 v_pca
  验证A: 内层预算 {5,15,30} × 外层上限 300 → 收敛次数与最终方向
  验证B: 不同内层预算下 d* 是否一致 (跨预算稳定性)
  验证C: angle(d*, v_pca) 是否小 (PCA 仅事后校验)
"""
import math
import time

import numpy as np

import n2sjy2 as n2


def fast_probs(r30, r45, t1, t2, a, rng=20):
    Z1 = np.asarray(r30, complex)
    Z2 = np.asarray(r45, complex)
    mg = np.arange(-rng, rng + 1)
    MM, NN = np.meshgrid(mg, mg)
    L1 = MM.ravel() + NN.ravel() * t1
    L2 = MM.ravel() + NN.ravel() * t2
    d1 = np.abs(Z1[:, None] - L1[None, :]).min(axis=1)
    d2 = np.abs(Z2[:, None] - L2[None, :]).min(axis=1)
    s1 = np.sqrt(np.mean(d1 ** 2))
    s2 = np.sqrt(np.mean(d2 ** 2))
    p1 = np.exp(-d1 ** 2 / (2 * s1 ** 2))
    p2 = np.exp(-d2 ** 2 / (2 * s2 ** 2))
    pt = np.exp(-np.abs(d1 + d2 - 2 * a) / (2 * a))
    return pt, p1, p2, np.array([2 * a, s1, s2])


n2.compute_probabilities_from_taus = fast_probs


def eval_res(x, pts, axis, r30, r45, a):
    return n2.evaluate_residuals(x, pts, axis, r30, r45, a,
                                 20.0, 1.0, 1e-3, 1.0, None, None)


def lm_step(x, pts, axis, r30, r45, a, lam=1e-3, fd_h=1e-3):
    r, F1, F2 = eval_res(x, pts, axis, r30, r45, a)
    cost = float(np.sum(r ** 2))
    J = np.zeros((len(r), 4))
    for k in range(4):
        xp = x.copy()
        xp[k] += fd_h
        rp2, _, _ = eval_res(xp, pts, axis, r30, r45, a)
        J[:, k] = (rp2 - r) / fd_h
    A = J.T @ J
    g = J.T @ r
    Aaug = A.copy()
    for c in range(4):
        Aaug[c, c] += lam * (A[c, c] + 1e-12)
    try:
        delta = np.linalg.solve(Aaug, g)
    except np.linalg.LinAlgError:
        return x, lam, F1, F2
    xtry = x - delta
    r2, _, _ = eval_res(xtry, pts, axis, r30, r45, a)
    cost2 = float(np.sum(r2 ** 2))
    if cost2 < cost:
        return xtry, max(lam / 3.0, 1e-8), F1, F2
    return x, min(lam * 3.0, 1e4), F1, F2


def inner_refine(t1, t2, pts, r30, r45, a, axis, n_inner, inner_tol=None):
    """内层: 热启动 LM 步; inner_tol 非空 = 连续 stall 次无进展(|Δx|<tol)才停(真正的内层极限)"""
    x = np.array([t1.real, t1.imag, t2.real, t2.imag])
    lam = 1e-3
    F1 = F2 = None
    stall = 0
    for _ in range(n_inner):
        x_prev = x.copy()
        x, lam, F1, F2 = lm_step(x, pts, axis, r30, r45, a, lam=lam)
        if inner_tol is not None:
            if np.max(np.abs(x - x_prev)) < inner_tol:
                stall += 1
                if stall >= 10:
                    break
            else:
                stall = 0
    return complex(x[0], x[1]), complex(x[2], x[3]), F1, F2


def data_driven_axis(points, d0, omega=0.9, n_inner=8, max_outer=200, tol_deg=0.5,
                     inner_tol=None, verbose=False):
    """无 PCA 方向不动点 (与 n2sjy2 外环同构, 但: 热启动 + 阻尼 + 方向稳定判据):
       τ ← 内层LM(方向目标=d, 热启动) → 自由重拟合焦点(extract_foci) → d_new → 阻尼
       判据: |d_new - d| < tol (方向不再变化, 无 PCA)"""
    rp = n2.compute_rp(points)
    d2 = math.asin(math.cos(math.radians(30)) / math.pi)
    a = rp * (1 + math.sin(d2))
    _, r45, r30, _ = n2.yzqx(points, rp)

    t1 = t2 = complex(0, 1)
    d = np.array(d0, float)
    d = d / np.linalg.norm(d)
    hist = []
    iters = 0
    for k in range(max_outer):
        # 1) 内层: 热启动 LM, 方向目标=当前 d
        t1, t2, F1, F2 = inner_refine(t1, t2, points, r30, r45, a, d, n_inner, inner_tol)
        # 2) 自由重拟合焦点 (无方向约束, 携带数据信息) — 用廉价 extract_foci
        pt, p1, p2, sig = fast_probs(r30, r45, t1, t2, a)
        F1f, F2f = n2.extract_foci(points, p1, sig[1], p2, sig[2])
        d_new = np.array(F1f, float) - np.array(F2f, float)
        if np.linalg.norm(d_new) < 1e-12:
            break
        d_new = d_new / np.linalg.norm(d_new)
        chg = math.degrees(math.acos(np.clip(np.dot(d_new, d), -1, 1)))
        # 3) 阻尼更新 (消除震荡)
        d = d + omega * (d_new - d)
        d = d / np.linalg.norm(d)
        hist.append(chg)
        iters = k + 1
        if chg < tol_deg:
            break
    return d, t1, t2, hist, iters


def init_direction(points):
    rp = n2.compute_rp(points)
    d2 = math.asin(math.cos(math.radians(30)) / math.pi)
    a = rp * (1 + math.sin(d2))
    _, r45, r30, _ = n2.yzqx(points, rp)
    t1t, t2t = n2.compute_taus()
    pt, p1, p2, sig = n2.compute_probabilities_from_taus(r30, r45, t1t, t2t, a)
    F1, F2 = n2.extract_foci(points, p1, sig[1], p2, sig[2])
    return np.array(F1) - np.array(F2)


def make_cube(seed):
    return np.random.RandomState(seed).uniform(-1, 1, (200, 3))


def make_ball(seed):
    rng = np.random.default_rng(200 + seed)
    dirs = rng.standard_normal((200, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return [d * (rng.random() ** (1 / 3)) for d in dirs]


def make_ellip(seed):
    rng = np.random.default_rng(300 + seed)
    u = rng.standard_normal(3)
    u /= np.linalg.norm(u)
    tmp = np.array([1.0, 0, 0]) if abs(u[0]) < 0.9 else np.array([0.0, 1, 0])
    v = np.cross(u, tmp)
    v /= np.linalg.norm(v)
    w = np.cross(u, v)
    th = rng.uniform(0, np.pi, 200)
    ph = rng.uniform(0, 2 * np.pi, 200)
    pts = []
    for i in range(200):
        pts.append(1.5 * np.sin(th[i]) * np.cos(ph[i]) * u
                   + 0.8 * np.sin(th[i]) * np.sin(ph[i]) * v
                   + 0.8 * np.cos(th[i]) * w + 0.02 * rng.standard_normal(3))
    return pts


def angle(u, v):
    return math.degrees(math.acos(np.clip(abs(np.dot(u, v)), -1, 1)))


def main():
    datasets = {}
    datasets["cube0"] = make_cube(10)
    datasets["cube1"] = make_cube(11)
    datasets["ball0"] = make_ball(0)
    datasets["ellip0"] = make_ellip(0)

    inner_opts = [5, 15]
    print("=" * 100)
    print("数据驱动无PCA方向不动点: 阻尼ω=0.9, 自由重拟合=extract_foci, 判据|Δd|<0.5°")
    print("内层预算 {5,15} 扫描 → 收敛次数 / angle(d*,v_pca) / 跨预算一致性")
    print("=" * 100)
    n_ok = 0
    for name, pts in datasets.items():
        d0 = init_direction(pts)
        v_pca = n2.pca_axis(pts)
        outs = {}
        for ni in inner_opts:
            d, _, _, hist, iters = data_driven_axis(pts, d0, n_inner=ni, max_outer=200)
            ang = angle(d, v_pca)
            outs[ni] = (iters, ang, d)
        ds = [outs[ni][2] for ni in inner_opts]
        spread = angle(ds[0], ds[1])
        ok = all(outs[ni][1] < 10 for ni in inner_opts)
        n_ok += 1 if ok else 0
        print(f"{name:8s} "
              + "".join(f"it={outs[ni][0]:3d} ang={outs[ni][1]:6.2f}°" for ni in inner_opts)
              + f"   跨预算Δd*={spread:5.2f}°  {'达标' if ok else '不达标'}")

    print()
    print(f"达标率 (全部内层预算下 angle(d*,v_pca)<10°): {n_ok}/{len(datasets)}")
    print("DONE")


if __name__ == "__main__":
    main()
