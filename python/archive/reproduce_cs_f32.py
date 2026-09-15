# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
float32 模拟 Unity (混合精度): lattice 距离 double → 存 float32 → 概率 float32 → fit float32
跑新版循环 (line81 pcav试探 + 热启t + NM + a9<10) on points.json, 看 a9 是否收敛 ~9.6
"""
import json
import math
import sys

import numpy as np

import n2sjy2 as n2
import nsjy_algorithms as m
import reproduce_cs_outer_faithful as cs
from reproduce_cs_outer_faithful import norm, angle, rotate_deg, normalize_tau_cs
from nelder_mead_cs import nelder_mead_cs

F32 = np.float32


def lattice_probs_f32(r30, r45, t1, t2, a, rng=20):
    """C# ComputeProbabilitiesFromTaus: lattice double, d 存 float, prob float"""
    Z1 = np.asarray(r30, complex)
    Z2 = np.asarray(r45, complex)
    mg = np.arange(-rng, rng + 1)
    MM, NN = np.meshgrid(mg, mg)
    L1 = MM.ravel() + NN.ravel() * t1
    L2 = MM.ravel() + NN.ravel() * t2
    d1 = np.abs(Z1[:, None] - L1[None, :]).min(axis=1).astype(F32)   # (float)d1
    d2 = np.abs(Z2[:, None] - L2[None, :]).min(axis=1).astype(F32)
    s1 = F32(math.sqrt(float(np.mean(d1.astype(np.float64) ** 2))))  # float sqrt
    s2 = F32(math.sqrt(float(np.mean(d2.astype(np.float64) ** 2))))
    p1 = np.exp(-(d1.astype(np.float64) ** 2) / (2.0 * float(s1) ** 2)).astype(F32)
    p2 = np.exp(-(d2.astype(np.float64) ** 2) / (2.0 * float(s2) ** 2)).astype(F32)
    pt = np.exp(-np.abs(d1.astype(np.float64) + d2.astype(np.float64) - 2.0 * float(a))
                / (2.0 * float(a))).astype(F32)
    return pt, p1, p2, np.array([F32(2 * a), s1, s2], F32)


def fit_foci_f32(P32, prob32, F1, F2, a, iters=300, lr=F32(0.0001), lambda_sep=F32(1.0), eps=F32(1e-3)):
    """n2sjy2 405 版全 float32 模拟 (Vector3 float)"""
    F1 = np.asarray(F1, F32).astype(F32).copy()
    F2 = np.asarray(F2, F32).astype(F32).copy()
    a = F32(a)
    for _ in range(iters):
        d1 = np.linalg.norm(P32 - F1, axis=1)
        d2 = np.linalg.norm(P32 - F2, axis=1)
        sd1 = np.maximum(d1, eps)
        sd2 = np.maximum(d2, eps)
        delta = d1 + d2 - F32(2) * a
        fp = np.exp(-np.abs(delta.astype(np.float64)) / (F32(2) * a).astype(np.float64)).astype(F32)
        diff = fp - prob32
        loss = float(np.sum(diff.astype(np.float64) ** 2)) + float(lambda_sep) * float(
            np.dot(F1 - F2, F1 - F2))
        if loss < 1e-12:
            break
        sign = np.where(delta >= 0, F32(1), F32(-1))
        coef = diff * fp * sign / (F32(2) * a)
        g1 = coef[:, None] * (P32 - F1) / sd1[:, None]
        g2 = coef[:, None] * (P32 - F2) / sd2[:, None]
        n1 = np.linalg.norm(g1, axis=1)
        n2 = np.linalg.norm(g2, axis=1)
        g1 = g1 * np.where(n1 > 1.0, F32(1) / n1, F32(1))[:, None]
        g2 = g2 * np.where(n2 > 1.0, F32(1) / n2, F32(1))[:, None]
        grad1 = g1.sum(0)
        grad2 = g2.sum(0)
        sep = F1 - F2
        grad1 = grad1 + F32(2) * lambda_sep * sep
        grad2 = grad2 - F32(2) * lambda_sep * sep
        gn1 = np.linalg.norm(grad1)
        gn2 = np.linalg.norm(grad2)
        if gn1 > 5.0:
            grad1 = grad1 * (F32(5) / gn1)
        if gn2 > 5.0:
            grad2 = grad2 * (F32(5) / gn2)
        F1 = F1 - lr * grad1
        F2 = F2 - lr * grad2
        if not (np.all(np.isfinite(F1)) and np.all(np.isfinite(F2))):
            break
    return F1, F2


def evaluate_f32(x, pts, P32, axis, r30, r45, a, t10, t20,
                 w_dir=F32(1.0), w_self=F32(1.0), w_theory=F32(1e-3)):
    n = len(pts)
    t1 = complex(x[0], x[1])
    t2 = complex(x[2], x[3])
    pt, p1, p2, sig = lattice_probs_f32(r30, r45, t1, t2, a)
    # ExtractFoci: dist = sigma*sqrt(-2ln p) float → lstsq double → Vector3 float
    def dists(prob, sigma):
        pp = np.clip(prob.astype(np.float64), 1e-6, 1.0 - 1e-6)
        return (float(sigma) * np.sqrt(-2.0 * np.log(pp))).astype(F32)
    dist1 = dists(p1, sig[1])
    dist2 = dists(p2, sig[2])
    F1 = m.fit_focus(pts, list(dist1))
    F2 = m.fit_focus(pts, list(dist2))
    F1 = F1.astype(F32)
    F2 = F2.astype(F32)
    f1, f2 = fit_foci_f32(P32, pt, F1, F2, a)
    r = np.zeros(n + 9)
    inv_sqrt_n = F32(1.0 / math.sqrt(n))
    d1 = np.linalg.norm(P32 - f1, axis=1)
    d2 = np.linalg.norm(P32 - f2, axis=1)
    delta = d1 + d2 - F32(2) * F32(a)
    prob_foci = np.exp(-np.abs(delta.astype(np.float64)) / (F32(2) * F32(a)).astype(np.float64)).astype(F32)
    r[:n] = (w_self * (pt - prob_foci) * inv_sqrt_n).astype(np.float64)
    dir_vec = F1 - F2
    if np.linalg.norm(dir_vec) < 1e-12:
        dir_vec = axis.astype(F32)
    dir_vec = dir_vec / np.linalg.norm(dir_vec)
    if np.dot(dir_vec, axis) < 0:
        dir_vec = -dir_vec
    e = dir_vec - axis
    r[n] = float(w_dir) * float(e[0])
    r[n + 1] = float(w_dir) * float(e[1])
    r[n + 2] = float(w_dir) * float(e[2])
    r[n + 3] = float(w_theory) * (x[0] - t10.real)
    r[n + 4] = float(w_theory) * (x[1] - t10.imag)
    r[n + 5] = float(w_theory) * (x[2] - t20.real)
    r[n + 6] = float(w_theory) * (x[3] - t20.imag)
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
    return r, F1.astype(np.float64), F2.astype(np.float64)


def refine_f32(pts, r30, r45, a, t10, t20, axis_v, max_iter=50,
               angle_tol_deg=F32(0.5), fd_h=F32(1e-3), capture_odr=False):
    P32 = np.asarray(pts, F32)
    axis = norm(np.asarray(axis_v, F32).astype(np.float64))
    x = np.array([t10.real, t10.imag, t20.real, t20.imag], float)
    lam = 1e-3
    captured = None
    F1 = F2 = np.zeros(3)
    for it in range(max_iter):
        r, F1, F2 = evaluate_f32(x, pts, P32, axis, r30, r45, a, t10, t20)
        cost = float(np.sum(r * r))
        dv = F1 - F2
        ang = 180.0
        if np.linalg.norm(dv) > 1e-12:
            dv = dv / np.linalg.norm(dv)
            if np.dot(dv, axis) < 0:
                dv = -dv
            ang = angle(dv, axis)
            if capture_odr and captured is None:
                captured = ang
        if ang < float(angle_tol_deg):
            break
        J = np.zeros((len(r), 4))
        for k in range(4):
            xp = x.copy()
            xp[k] += float(fd_h)
            xp = normalize_tau_cs(xp)
            rp2, _, _ = evaluate_f32(xp, pts, P32, axis, r30, r45, a, t10, t20)
            J[:, k] = (rp2 - r) / float(fd_h)
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
        r2, _, _ = evaluate_f32(xtry, pts, P32, axis, r30, r45, a, t10, t20)
        cost2 = float(np.sum(r2 * r2))
        if cost2 < cost:
            x = xtry
            lam = max(lam / 3.0, 1e-8)
        else:
            lam = min(lam * 3.0, 1e4)
    r, F1, F2 = evaluate_f32(x, pts, P32, axis, r30, r45, a, t10, t20)
    return (complex(x[0], x[1]), complex(x[2], x[3]), F1, F2, captured)


def main():
    with open(r"C:\Users\23128\My project (2)\Assets\Resources\points.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    pts = [np.array(p, float) for p in raw]
    P = np.array(pts, float)
    P32 = P.astype(F32)
    rp = n2.compute_rp(pts)
    d2 = math.asin(math.cos(math.radians(30)) / math.pi)
    a = rp * (1 + math.sin(d2))
    _, r45, r30, _ = n2.yzqx(pts, rp)
    t1t, t2t = n2.compute_taus()
    pc1, v3, _ = m.pca(pts)
    v3 = norm(v3)
    C = P.T @ P
    evals, evecs = np.linalg.eigh(C)
    v_true = norm(evecs[:, np.argsort(evals)[::-1][0]])
    t_ = complex(0, 1)

    # 初始 f1/f2 (float32 fit)
    pt0, p10, p20, sg0 = lattice_probs_f32(r30, r45, t1t, t2t, a)
    dist1 = (float(sg0[1]) * np.sqrt(-2.0 * np.log(np.clip(p10.astype(np.float64), 1e-6, 1 - 1e-6)))).astype(F32)
    dist2 = (float(sg0[2]) * np.sqrt(-2.0 * np.log(np.clip(p20.astype(np.float64), 1e-6, 1 - 1e-6)))).astype(F32)
    F1i = m.fit_focus(pts, list(dist1)).astype(F32)
    F2i = m.fit_focus(pts, list(dist2)).astype(F32)
    f1, f2 = fit_foci_f32(P32, pt0, F1i, F2i, a)
    # f01/f02 ((0,1))
    pt01, p101, p201, sg01 = lattice_probs_f32(r30, r45, t_, t_, a)
    d101 = (float(sg01[1]) * np.sqrt(-2.0 * np.log(np.clip(p101.astype(np.float64), 1e-6, 1 - 1e-6)))).astype(F32)
    d201 = (float(sg01[2]) * np.sqrt(-2.0 * np.log(np.clip(p201.astype(np.float64), 1e-6, 1 - 1e-6)))).astype(F32)
    F01i = m.fit_focus(pts, list(d101)).astype(F32)
    F02i = m.fit_focus(pts, list(d201)).astype(F32)
    f01, f02 = fit_foci_f32(P32, pt01, F01i, F02i, a)
    v4 = norm((f01 - f02).astype(np.float64))
    v2 = norm((f1 - f2).astype(np.float64))

    print(f"float32 模拟 points.json  ∠(v3,真轴)={angle(v3, v_true):.2f}°")
    print(f"  v2∠v3={angle(v2, v3):.2f}°  v4∠v3={angle(v4, v3):.2f}°")
    # line 81: 直接对 pcav 试探 (结果未用)
    t1n_, t2n_, F1n_, F2n_, odr_ = refine_f32(pts, r30, r45, a, t_, t_, v3, max_iter=50)
    print(f"  [line81 pcav试探] t1={t1n_} t2={t2n_}")
    # 循环
    t1 = t2 = complex(0, 1)
    f1, f2 = f1.astype(np.float64), f2.astype(np.float64)
    print(f"  {'it':>3s} {'a9(v3)':>8s} {'a5':>7s} {'odr':>7s} {'new':>7s} {'stop?':>5s}")
    for i in range(100):
        axi = norm(f1 - f2)
        ang_new = angle(axi, v4)
        t1n, t2n, F1n, F2n, odr = refine_f32(pts, r30, r45, a, t1, t2, axi, max_iter=50, capture_odr=True)
        odr = odr if odr is not None else 0.0
        ptn, p1n, p2n, sgn = lattice_probs_f32(r30, r45, t1n, t2n, a)
        dn1 = (float(sgn[1]) * np.sqrt(-2.0 * np.log(np.clip(p1n.astype(np.float64), 1e-6, 1 - 1e-6)))).astype(F32)
        dn2 = (float(sgn[2]) * np.sqrt(-2.0 * np.log(np.clip(p2n.astype(np.float64), 1e-6, 1 - 1e-6)))).astype(F32)
        F1x = m.fit_focus(pts, list(dn1)).astype(F32)
        F2x = m.fit_focus(pts, list(dn2)).astype(F32)
        f1n, f2n = fit_foci_f32(P32, ptn, F1x, F2x, a)
        v6 = norm((f1n - f2n).astype(np.float64))
        a9 = angle(v6, v3)
        a5 = angle(v6, v2)
        flag = a9 < 10
        if i % 2 == 0 or flag:
            print(f"  {i:3d} {a9:8.3f} {a5:7.2f} {odr:7.2f} {ang_new:7.2f} {str(flag):>5s}", flush=True)
        if flag:
            print(f"  ==> 收敛! a9={a9:.6f} a5={a5:.6f} new={ang_new:.6f} odr={odr:.6f} @iter{i}")
            break
        # 旋转 + NM (C# NM 精确)
        ang5 = angle(v6, v2)
        ang6 = angle(v6, v4)
        axis = np.cross(v6, v4)
        axis1 = np.cross(v6, v2)
        newf = norm(cs.rotate_deg(v4, axis, -ang5))
        newv = norm(cs.rotate_deg(v2, axis1, -ang6))
        lb = np.array([-0.5, 0.5, -0.5, 0.5])
        ub = np.array([0.5, 2.0, 0.5, 2.0])
        st = {"F1z": f1n.astype(np.float64), "F2z": f2n.astype(np.float64)}

        def obj(x):
            tt1 = complex(x[0], x[1])
            tt2 = complex(x[2], x[3])
            ptc, pc1, pc2, sc = lattice_probs_f32(r30, r45, tt1, tt2, a)
            dc1 = (float(sc[1]) * np.sqrt(-2.0 * np.log(np.clip(pc1.astype(np.float64), 1e-6, 1 - 1e-6)))).astype(F32)
            dc2 = (float(sc[2]) * np.sqrt(-2.0 * np.log(np.clip(pc2.astype(np.float64), 1e-6, 1 - 1e-6)))).astype(F32)
            Fc1 = m.fit_focus(pts, list(dc1)).astype(F32)
            Fc2 = m.fit_focus(pts, list(dc2)).astype(F32)
            fc1, fc2 = fit_foci_f32(P32, ptc, Fc1, Fc2, a)
            st["F1z"], st["F2z"] = fc1.astype(np.float64), fc2.astype(np.float64)
            dv = fc1 - fc2
            if np.dot(dv, dv) < 1e-12:
                return 0.0
            dv = dv / np.linalg.norm(dv)
            return abs(angle(dv, newf) - angle(dv, newv))

        x0 = np.clip([t1n.real, t1n.imag, t2n.real, t2n.imag], lb, ub)
        bx, bf, nev = nelder_mead_cs(obj, x0, lb, ub, max_evals=100, stop_f=16.0, tol=1e-6)
        t1, t2 = complex(bx[0], bx[1]), complex(bx[2], bx[3])
        f1, f2 = st["F1z"], st["F2z"]


if __name__ == "__main__":
    main()
