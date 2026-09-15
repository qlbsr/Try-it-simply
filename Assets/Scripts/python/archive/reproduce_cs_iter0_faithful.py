# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
逐行对齐 C# n2sjy2 的 iter0 (pyjson), 对照 Unity 日志:
    (angleDeg9=2.002618, angleDegnew=111.2141, angleDegodr=70.66569)
    [iter 0] cost=... angle=70.666° t1=(0,1) t2=(0,1)   (RefineModuliByAxis 内)

与旧 reproduce_cs_pyjson.py 的差异 (全部按 C# nsjy.cs / n2sjy2.cs 修正):
  1. fit_foci: lr=0.001, d>1e-6 才贡献梯度(否则跳过), 无 sep 正则/无梯度裁剪 (C# nsjy.cs 206-251)
  2. EvaluateResiduals 概率自洽项: 先 FitFociByProbability(probs[0]) 得 f1,f2,
     再用拟合后 f1,f2 反推 probFoci (C# n2sjy2.cs 639-656)  ← 旧版用未拟合 F1,F2
  3. dir-closure 项用 ExtractFoci 的 F1-F2 (C# 674), wDir=1 (C# 550 默认)
  4. 含理论正则 wTheory=1e-3 相对 t10,t20=(0,1) (r[n+3..n+6]) 与 dy*(Im-1) 项 (r[n+7..n+8])
  5. NormalizeTau 无扰动 (C# 702-730)
  6. 主循环只在接受 xtry 时 normalize; x 初值 (0,1,0,1) 不预规范化
"""
import json
import math
import time

import numpy as np

import n2sjy2 as n2
import nsjy_algorithms as m

PYJSON = r"C:\Users\23128\My project (2)\Assets\Resources\pyjson.json"


def angle(u, vv):
    return math.degrees(math.acos(np.clip(np.dot(u, vv) / (np.linalg.norm(u) * np.linalg.norm(vv)), -1, 1)))


def norm(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


# ---------- 1. FitFociByProbability 忠实版 (向量化, 语义= C# nsjy.cs) ----------
def fit_foci_cs(P, prob, F1, F2, a, iters=300, lr=0.001):
    F1 = np.asarray(F1, float).copy()
    F2 = np.asarray(F2, float).copy()
    for _ in range(iters):
        d1 = np.linalg.norm(P - F1, axis=1)
        d2 = np.linalg.norm(P - F2, axis=1)
        delta = d1 + d2 - 2.0 * a
        fp = np.exp(-np.abs(delta) / (2.0 * a))
        diff = fp - prob
        loss = float(np.sum(diff * diff))
        if loss < 1e-12:
            break
        sign = np.where(delta >= 0, 1.0, -1.0)
        coef = diff * fp * sign / (2.0 * a)
        g1 = np.zeros(3)
        g2 = np.zeros(3)
        m1 = d1 > 1e-6
        m2 = d2 > 1e-6
        if m1.any():
            g1 = np.sum(coef[m1, None] * (P[m1] - F1) / d1[m1, None], axis=0)
        if m2.any():
            g2 = np.sum(coef[m2, None] * (P[m2] - F2) / d2[m2, None], axis=0)
        F1 -= lr * g1
        F2 -= lr * g2
    return F1, F2


# ---------- 2. fast lattice probs (与 C# ComputeProbabilitiesFromTaus 相同) ----------
def lattice_probs(r30, r45, t1, t2, a, rng=20):
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


# ---------- 3. NormalizeTau 忠实 (无扰动) ----------
def normalize_tau_cs(x):
    x = x.copy()
    for k in range(0, 4, 2):
        re, im = x[k], x[k + 1]
        for _ in range(100):
            re -= round(re)
            mod2 = re * re + im * im
            if mod2 >= 1.0 - 1e-14:
                break
            den = mod2
            re, im = -re / den, im / den
        x[k], x[k + 1] = re, im
    return x


# ---------- 4. EvaluateResiduals 忠实 (n+9, C# 629-698) ----------
def evaluate_residuals_cs(x, pts, axis, r30, r45, a, t10, t20,
                          w_dir=1.0, w_self=1.0, w_theory=1e-3):
    P = np.asarray(pts, float)
    n = len(pts)
    t1 = complex(x[0], x[1])
    t2 = complex(x[2], x[3])
    pt, p1, p2, sig = lattice_probs(r30, r45, t1, t2, a)
    # ExtractFoci (C# 639)
    F1, F2 = m.extract_foci(pts, p1, sig[1], p2, sig[2])
    # FitFociByProbability (C# 640) → 拟合后焦点
    f1, f2 = fit_foci_cs(P, pt, F1, F2, a)
    r = np.zeros(n + 9)
    inv_sqrt_n = 1.0 / math.sqrt(n)
    # 概率自洽: 拟合后 f1,f2 反推 probFoci (C# 648-656)
    d1 = np.linalg.norm(P - f1, axis=1)
    d2 = np.linalg.norm(P - f2, axis=1)
    delta = d1 + d2 - 2.0 * a
    prob_foci = np.exp(-np.abs(delta) / (2.0 * a))
    r[:n] = w_self * (pt - prob_foci) * inv_sqrt_n
    # 方向闭合: ExtractFoci 的 F1-F2 (C# 674-681)
    dir_vec = F1 - F2
    if np.linalg.norm(dir_vec) < 1e-12:
        dir_vec = axis
    dir_vec = dir_vec / np.linalg.norm(dir_vec)
    if np.dot(dir_vec, axis) < 0:
        dir_vec = -dir_vec
    e = dir_vec - axis
    r[n] = w_dir * e[0]
    r[n + 1] = w_dir * e[1]
    r[n + 2] = w_dir * e[2]
    # 理论正则 (C# 690-693)
    r[n + 3] = w_theory * (x[0] - t10.real)
    r[n + 4] = w_theory * (x[1] - t10.imag)
    r[n + 5] = w_theory * (x[2] - t20.real)
    r[n + 6] = w_theory * (x[3] - t20.imag)
    # dy*(Im-1) 拉向 Im=1 (C# 658-697)
    im1 = x[1]
    dev1 = 0
    for i in range(100):
        if im1 - 2 ** dev1 < 0:
            break
        dev1 += 1
    dy = 1.0
    for i in range(int(dev1)):
        dy += 1.0 / (2 ** i)
    r[n + 7] = dy * (x[1] - 1)
    r[n + 8] = dy * (x[3] - 1)
    return r, F1, F2, f1, f2


# ---------- 5. RefineModuliByAxis 忠实 LM (C# 571-626) ----------
def refine_moduli_cs(pts, r30, r45, a, t10, t20, axis_v,
                     max_iter=50, angle_tol_deg=0.5, fd_h=1e-3,
                     capture_odr=False, odr_state=0.0, verbose=False):
    axis = norm(axis_v)
    x = np.array([t10.real, t10.imag, t20.real, t20.imag], float)
    lam = 1e-3
    captured = None
    F1 = F2 = np.zeros(3)
    for it in range(max_iter):
        r, F1, F2, f1, f2 = evaluate_residuals_cs(x, pts, axis, r30, r45, a, t10, t20)
        cost = float(np.sum(r * r))
        dir_vec = F1 - F2
        angle_deg = 180.0
        if np.linalg.norm(dir_vec) > 1e-12:
            dir_vec = dir_vec / np.linalg.norm(dir_vec)
            if np.dot(dir_vec, axis) < 0:
                dir_vec = -dir_vec
            angle_deg = angle(dir_vec, axis)
            # C# 583-586: angleDegodr==0 首次捕获 (外环 reset=0)
            if capture_odr and odr_state == 0.0 and captured is None:
                captured = angle_deg
        if verbose:
            print(f"[iter {it}] cost={cost:.3E} angle={angle_deg:.3f}° "
                  f"t1=({x[0]:.4f},{x[1]:.4f}) t2=({x[2]:.4f},{x[3]:.4f})")
        if angle_deg < angle_tol_deg:
            if verbose:
                print("闭环收敛")
            break
        J = np.zeros((len(r), 4))
        for k in range(4):
            xp = x.copy()
            xp[k] += fd_h
            xp = normalize_tau_cs(xp)          # C# 597: NormalizeTau(xp) 无扰动
            rp2, _, _, _, _ = evaluate_residuals_cs(xp, pts, axis, r30, r45, a, t10, t20)
            J[:, k] = (rp2 - r) / fd_h
        A = J.T @ J
        g = J.T @ r
        A_aug = A.copy()
        for c in range(4):
            A_aug[c, c] += lam * (A[c, c] + 1e-12)
        try:
            delta = np.linalg.solve(A_aug, g)
        except np.linalg.LinAlgError:
            break
        xtry = x - delta
        xtry = normalize_tau_cs(xtry)          # C# 617
        r2, _, _, _, _ = evaluate_residuals_cs(xtry, pts, axis, r30, r45, a, t10, t20)
        cost2 = float(np.sum(r2 * r2))
        if cost2 < cost:
            x = xtry
            lam = max(lam / 3.0, 1e-8)
        else:
            lam = min(lam * 3.0, 1e4)
    r, F1, F2, f1, f2 = evaluate_residuals_cs(x, pts, axis, r30, r45, a, t10, t20)
    return (complex(x[0], x[1]), complex(x[2], x[3]), F1, F2, captured)


# ---------- 主流程: C# n2sjy2 Start + 外环 iter0 ----------
def main():
    with open(PYJSON, "r", encoding="utf-8") as f:
        raw = json.load(f)
    pts = [np.array([p["x"], p["y"], p["z"]], float) for p in raw]
    P = np.array(pts, float)
    n = len(pts)
    rp = n2.compute_rp(pts)
    d2 = math.asin(math.cos(math.radians(30)) / math.pi)
    a = rp * (1 + math.sin(d2))
    e3 = math.cos(d2) * 2 / (1 + math.sin(d2))
    c = rp * math.cos(d2) * e3
    _, r45, r30, _ = n2.yzqx(pts, rp)
    t1t, t2t = n2.compute_taus()               # ComputeTaus(2, sqrt2)
    t_ = complex(0, 1)
    v3 = norm(m.pca(pts)[1])                   # C# pca(points).normalized (v3 重建)
    C = P.T @ P
    evals, evecs = np.linalg.eigh(C)
    v_true = norm(evecs[:, np.argsort(evals)[::-1][0]])

    print(f"pyjson n={n} rp={rp:.4f} a={a:.4f}")
    print(f"t1t={t1t}  t2t={t2t}")

    # --- C# line 61-63: probs12(t1t,t2t) → Extract → fit f1,f2 ---
    pt12, p112, p212, sig12 = lattice_probs(r30, r45, t1t, t2t, a)
    F1, F2 = m.extract_foci(pts, p112, sig12[1], p212, sig12[2])
    f1, f2 = fit_foci_cs(P, pt12, F1, F2, a)
    print(f"\n[初始拟合 真taus] F1={F1} F2={F2}")
    print(f"  fit后 f1={f1} f2={f2}  ∠(f1-f2, v3)={angle(norm(f1-f2), v3):.3f}")

    # --- C# line 68-70: probs012(t_,t_) → Extract → fit f01,f02 ---
    pt01, p101, p201, sig01 = lattice_probs(r30, r45, t_, t_, a)
    F01, F02 = m.extract_foci(pts, p101, sig01[1], p201, sig01[2])
    f01, f02 = fit_foci_cs(P, pt01, F01, F02, a)
    print(f"\n[初始拟合 (0,1)] F01={F01} F02={F02}")
    print(f"  fit后 f01={f01} f02={f02}")

    # --- 外环 iter0 (C# 81-153) ---
    # line 85: angleDegnew = ∠(f1-f2, f01-f02)  (f01/f02 永不更新)
    ang_new = angle(norm(f1 - f2), norm(f01 - f02))
    ang_new2 = angle(norm(f1 - f2), v3)     # anglenew line 87
    print(f"\n[iter0] line85 angleDegnew = ∠(f1-f2,f01-f02) = {ang_new:.4f}  (Unity 111.2141)")
    print(f"[iter0] line87 anglenew     = ∠(f1-f2,v3)      = {ang_new2:.4f}")
    d = norm(f1 - f2)
    r_dir = norm(f01 - f02)
    axi = d
    if ang_new < 5:
        n_ax = np.cross(d, r_dir)
        if np.linalg.norm(n_ax) < 1e-12:
            n_ax = np.cross(d, np.array([0., 1., 0.]))
        axi = norm(rotate_deg(d, n_ax, 60.0))
        print(f"  <5° 扰动: axi = rotate60(d)   (iter0 实际不会触发: {ang_new:.1f}°>5°)")

    # line 123: RefineModuliByAxis(points, t_, t_, axi, 50)
    t0 = time.time()
    t1n, t2n, F1n, F2n, odr = refine_moduli_cs(
        pts, r30, r45, a, t_, t_, axi, max_iter=50,
        capture_odr=True, odr_state=0.0, verbose=True)
    print(f"\n[iter0] RefineModuliByAxis 完成 ({time.time()-t0:.1f}s)")
    print(f"  t1n={t1n}  t2n={t2n}  odr(captured)={odr if odr is not None else 0.0:.4f}  (Unity 70.66569)")
    print(f"  F1n={F1n}  F2n={F2n}")

    # line 124-126: prob_new → Extract → fit
    ptn, p1n, p2n, sign = lattice_probs(r30, r45, t1n, t2n, a)
    F1x, F2x = m.extract_foci(pts, p1n, sign[1], p2n, sign[2])
    f1n, f2n = fit_foci_cs(P, ptn, F1x, F2x, a)
    v6 = norm(f1n - f2n)
    a9v3 = angle(v6, v3)
    a9true = angle(v6, v_true)
    print(f"\n[iter0] refit 后 v6 = (f1n-f2n) 归一化")
    print(f"  a9 = ∠(v6, v3)   = {a9v3:.4f}   (Unity 2.002618)")
    print(f"  a9 = ∠(v6, 真轴) = {a9true:.4f}")
    print(f"\n  核对: odr={odr if odr is not None else 0.0:.4f}  new={ang_new:.4f}  |odr-new|={abs((odr or 0.0)-ang_new):.4f}  (>16? {abs((odr or 0.0)-ang_new)>16})")
    print(f"  a9(v3)={a9v3:.4f}  → && a9<1 终止条件 {'触发' if abs((odr or 0.0)-ang_new)>16 and a9v3<1 else '不触发'}")


def rotate_deg(v, ax, deg):
    ax = np.asarray(ax, float)
    ax = ax / np.linalg.norm(ax)
    rg = math.radians(deg)
    return (v * math.cos(rg) + np.cross(ax, v) * math.sin(rg)
            + ax * np.dot(ax, v) * (1 - math.cos(rg)))


if __name__ == "__main__":
    main()
