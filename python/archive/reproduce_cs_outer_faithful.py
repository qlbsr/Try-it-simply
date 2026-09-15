# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
完整忠实外环复现 C# n2sjy2 (pyjson) - n2 版拟合 + 修正 v3
支持 RefineModuliByAxis 权重参数 (w_dir/w_self/w_theory) — 测试 wSelf 2.5 收敛性
"""
import json
import math
import sys

import numpy as np
from scipy.optimize import minimize

import n2sjy2 as n2
import nsjy_algorithms as m

PYJSON = r"C:\Users\23128\My project (2)\Assets\Resources\pyjson.json"


def norm(v):
    return np.asarray(v, float) / np.linalg.norm(v)


def angle(u, vv):
    return math.degrees(math.acos(np.clip(np.dot(u, vv) / (np.linalg.norm(u) * np.linalg.norm(vv)), -1, 1)))


def rotate_deg(v, ax, deg):
    ax = np.asarray(ax, float)
    nrm = np.linalg.norm(ax)
    if nrm < 1e-12:
        return norm(v)
    ax = ax / nrm
    rg = math.radians(deg)
    return (v * math.cos(rg) + np.cross(ax, v) * math.sin(rg)
            + ax * np.dot(ax, v) * (1 - math.cos(rg)))


# ---------- fit (n2sjy2.cs 405-469: lr=0.0001 + lambdaSep=1.0 + 裁剪) ----------
def fit_foci_n2(P, prob, F1, F2, a, iters=300, lr=0.0001, lambda_sep=1.0, eps=1e-3):
    F1 = np.asarray(F1, float).copy()
    F2 = np.asarray(F2, float).copy()
    for _ in range(iters):
        d1 = np.linalg.norm(P - F1, axis=1)
        d2 = np.linalg.norm(P - F2, axis=1)
        sd1 = np.maximum(d1, eps)
        sd2 = np.maximum(d2, eps)
        delta = d1 + d2 - 2.0 * a
        fp = np.exp(-np.abs(delta) / (2.0 * a))
        diff = fp - prob
        loss = float(np.sum(diff * diff)) + lambda_sep * float(np.dot(F1 - F2, F1 - F2))
        if loss < 1e-12:
            break
        sign = np.where(delta >= 0, 1.0, -1.0)
        coef = diff * fp * sign / (2.0 * a)
        g_raw1 = coef[:, None] * (P - F1) / sd1[:, None]
        g_raw2 = coef[:, None] * (P - F2) / sd2[:, None]
        n1 = np.linalg.norm(g_raw1, axis=1)
        n2 = np.linalg.norm(g_raw2, axis=1)
        g_raw1 = g_raw1 * np.where(n1 > 1.0, 1.0 / n1, 1.0)[:, None]
        g_raw2 = g_raw2 * np.where(n2 > 1.0, 1.0 / n2, 1.0)[:, None]
        grad1 = g_raw1.sum(0)
        grad2 = g_raw2.sum(0)
        sep = F1 - F2
        grad1 += 2.0 * lambda_sep * sep
        grad2 -= 2.0 * lambda_sep * sep
        gn1 = np.linalg.norm(grad1)
        gn2 = np.linalg.norm(grad2)
        if gn1 > 5.0:
            grad1 *= 5.0 / gn1
        if gn2 > 5.0:
            grad2 *= 5.0 / gn2
        F1 -= lr * grad1
        F2 -= lr * grad2
        if not (np.all(np.isfinite(F1)) and np.all(np.isfinite(F2))):
            break
    return F1, F2


# ---------- lattice 概率 (C# ComputeProbabilitiesFromTaus 相同) ----------
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


def evaluate_residuals_cs(x, pts, axis, r30, r45, a, t10, t20,
                          w_dir=1.0, w_self=1.0, w_theory=1e-3):
    P = np.asarray(pts, float)
    n = len(pts)
    t1 = complex(x[0], x[1])
    t2 = complex(x[2], x[3])
    pt, p1, p2, sig = lattice_probs(r30, r45, t1, t2, a)
    F1, F2 = m.extract_foci(pts, p1, sig[1], p2, sig[2])
    f1, f2 = fit_foci_n2(P, pt, F1, F2, a)
    r = np.zeros(n + 9)
    inv_sqrt_n = 1.0 / math.sqrt(n)
    d1 = np.linalg.norm(P - f1, axis=1)
    d2 = np.linalg.norm(P - f2, axis=1)
    delta = d1 + d2 - 2.0 * a
    prob_foci = np.exp(-np.abs(delta) / (2.0 * a))
    r[:n] = w_self * (pt - prob_foci) * inv_sqrt_n
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
    r[n + 3] = w_theory * (x[0] - t10.real)
    r[n + 4] = w_theory * (x[1] - t10.imag)
    r[n + 5] = w_theory * (x[2] - t20.real)
    r[n + 6] = w_theory * (x[3] - t20.imag)
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
    return r, F1, F2


def refine_moduli_cs(pts, r30, r45, a, t10, t20, axis_v, max_iter=50,
                     angle_tol_deg=0.5, fd_h=1e-3, capture_odr=False, verbose=False,
                     w_dir=1.0, w_self=1.0, w_theory=1e-3):
    axis = norm(axis_v)
    x = np.array([t10.real, t10.imag, t20.real, t20.imag], float)
    lam = 1e-3
    captured = None
    F1 = F2 = np.zeros(3)
    for it in range(max_iter):
        r, F1, F2 = evaluate_residuals_cs(x, pts, axis, r30, r45, a, t10, t20,
                                          w_dir=w_dir, w_self=w_self, w_theory=w_theory)
        cost = float(np.sum(r * r))
        dir_vec = F1 - F2
        angle_deg = 180.0
        if np.linalg.norm(dir_vec) > 1e-12:
            dir_vec = dir_vec / np.linalg.norm(dir_vec)
            if np.dot(dir_vec, axis) < 0:
                dir_vec = -dir_vec
            angle_deg = angle(dir_vec, axis)
            if capture_odr and captured is None:
                captured = angle_deg
        if verbose:
            print(f"[iter {it}] cost={cost:.3E} angle={angle_deg:.3f} t1=({x[0]:.4f},{x[1]:.4f}) t2=({x[2]:.4f},{x[3]:.4f})")
        if angle_deg < angle_tol_deg:
            break
        J = np.zeros((len(r), 4))
        for k in range(4):
            xp = x.copy()
            xp[k] += fd_h
            xp = normalize_tau_cs(xp)
            rp2, _, _ = evaluate_residuals_cs(xp, pts, axis, r30, r45, a, t10, t20,
                                              w_dir=w_dir, w_self=w_self, w_theory=w_theory)
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
        xtry = normalize_tau_cs(xtry)
        r2, _, _ = evaluate_residuals_cs(xtry, pts, axis, r30, r45, a, t10, t20,
                                         w_dir=w_dir, w_self=w_self, w_theory=w_theory)
        cost2 = float(np.sum(r2 * r2))
        if cost2 < cost:
            x = xtry
            lam = max(lam / 3.0, 1e-8)
        else:
            lam = min(lam * 3.0, 1e4)
    r, F1, F2 = evaluate_residuals_cs(x, pts, axis, r30, r45, a, t10, t20,
                                      w_dir=w_dir, w_self=w_self, w_theory=w_theory)
    return (complex(x[0], x[1]), complex(x[2], x[3]), F1, F2, captured)


# ---------- RefineTausWithNM (scipy 近似) ----------
class _Stop(Exception):
    pass


def nm_refine_cs(pts, r30, r45, a, t1i, t2i, ref_newf, ref_newv,
                 max_evals=100, stop_f=16.0):
    P = np.asarray(pts, float)
    lb = np.array([-0.5, 0.5, -0.5, 0.5])
    ub = np.array([0.5, 2.0, 0.5, 2.0])
    state = {"F1z": None, "F2z": None}

    def obj(x):
        tt1 = complex(x[0], x[1])
        tt2 = complex(x[2], x[3])
        ptc, pc1, pc2, sc = lattice_probs(r30, r45, tt1, tt2, a)
        Fc1, Fc2 = m.extract_foci(pts, pc1, sc[1], pc2, sc[2])
        fc1, fc2 = fit_foci_n2(P, ptc, Fc1, Fc2, a)
        state["F1z"], state["F2z"] = fc1, fc2
        d = fc1 - fc2
        if np.dot(d, d) < 1e-12:
            return 0.0
        d = d / np.linalg.norm(d)
        ad = angle(d, ref_newf)
        ad1 = angle(d, ref_newv)
        return abs(ad - ad1)

    x0 = np.clip([t1i.real, t1i.imag, t2i.real, t2i.imag], lb, ub)
    best_x = x0

    def cb(xk):
        nonlocal best_x
        best_x = xk.copy()
        if obj(xk) <= stop_f:
            raise _Stop()

    try:
        minimize(obj, x0, method="Nelder-Mead", bounds=list(zip(lb, ub)),
                 callback=cb, options={"maxiter": max_evals, "maxfev": max_evals * 2,
                                       "xatol": 1e-6, "fatol": 1e-6})
        best_x = np.clip(best_x, lb, ub)
    except _Stop:
        pass
    best_x = np.clip(best_x, lb, ub)
    val = obj(best_x)
    return (complex(best_x[0], best_x[1]), complex(best_x[2], best_x[3]),
            val, state["F1z"], state["F2z"])


# ---------- 数据加载 ----------
def load_data(path=PYJSON):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw[0], dict):
        pts = [np.array([p["x"], p["y"], p["z"]], float) for p in raw]
    else:
        pts = [np.array(p, float) for p in raw]
    P = np.array(pts, float)
    rp = n2.compute_rp(pts)
    d2 = math.asin(math.cos(math.radians(30)) / math.pi)
    a = rp * (1 + math.sin(d2))
    _, r45, r30, _ = n2.yzqx(pts, rp)
    pc1, v3, _ = m.pca(pts)
    evals, evecs = np.linalg.eigh(P.T @ P)
    v_true = norm(evecs[:, np.argsort(evals)[::-1][0]])
    return dict(pts=pts, P=P, r30=r30, r45=r45, a=a, v3=norm(v3), v_true=v_true)


def run(path=PYJSON, max_outer=8, use_cond2=False, tag="",
        w_dir=1.0, w_self=1.0, w_theory=1e-3):
    d = load_data(path)
    pts, P, r30, r45, a = d["pts"], d["P"], d["r30"], d["r45"], d["a"]
    v3, v_true = d["v3"], d["v_true"]
    t_ = complex(0, 1)
    t1t, t2t = n2.compute_taus()
    pt12, p112, p212, sig12 = lattice_probs(r30, r45, t1t, t2t, a)
    F1, F2 = m.extract_foci(pts, p112, sig12[1], p212, sig12[2])
    f1, f2 = fit_foci_n2(P, pt12, F1, F2, a)
    pt01, p101, p201, sig01 = lattice_probs(r30, r45, t_, t_, a)
    F01, F02 = m.extract_foci(pts, p101, sig01[1], p201, sig01[2])
    f01, f02 = fit_foci_n2(P, pt01, F01, F02, a)
    v2 = norm(f1 - f2)
    v4 = norm(f01 - f02)
    print(f"{tag}完整外环 (w_dir={w_dir}, w_self={w_self}) v3角真轴={angle(v3, v_true):.2f}")
    print(f"  {'iter':>4s} {'a9(v3)':>8s} {'a9真':>8s} {'odr':>7s} {'new':>7s} {'|D|':>7s}")
    stop = None
    for i in range(max_outer):
        d_cur = norm(f1 - f2)
        ang_new = angle(d_cur, v4)
        axi = d_cur
        if ang_new < 5:
            n_ax = np.cross(d_cur, v4)
            if np.linalg.norm(n_ax) < 1e-12:
                n_ax = np.cross(d_cur, np.array([0., 1., 0.]))
            axi = norm(rotate_deg(d_cur, n_ax, 60.0))
        t1n, t2n, F1n, F2n, odr = refine_moduli_cs(
            pts, r30, r45, a, t_, t_, axi, max_iter=50, capture_odr=True,
            w_dir=w_dir, w_self=w_self, w_theory=w_theory)
        odr = odr if odr is not None else 0.0
        ptn, p1n, p2n, sgn = lattice_probs(r30, r45, t1n, t2n, a)
        F1x, F2x = m.extract_foci(pts, p1n, sgn[1], p2n, sgn[2])
        f1n, f2n = fit_foci_n2(P, ptn, F1x, F2x, a)
        v6 = norm(f1n - f2n)
        a9v = angle(v6, v3)
        a9t = angle(v6, v_true)
        diff = abs(odr - ang_new)
        cond = diff > 16
        stop_flag = (cond and a9v < 1) if use_cond2 else cond
        print(f"  {i:4d} {a9v:8.2f} {a9t:8.2f} {odr:7.2f} {ang_new:7.2f} {diff:7.2f}", flush=True)
        if stop_flag:
            stop = i
            break
        ang5 = angle(v6, v2)
        ang6 = angle(v6, v4)
        axis = np.cross(v6, v4)
        axis1 = np.cross(v6, v2)
        newf = norm(rotate_deg(v4, axis, -ang5))
        newv = norm(rotate_deg(v2, axis1, -ang6))
        t1_, t2_, delta, F1z, F2z = nm_refine_cs(pts, r30, r45, a, t1n, t2n, newf, newv)
        f1, f2 = F1z, F2z
    print(f"  ==> stop@{stop} | 末轮 a9(v3)={a9v:.2f}")
    return a9v


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else PYJSON
    mo = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    run(path=p, max_outer=mo, use_cond2=False, tag="[A] ")
