# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
从概率约束项中找使 n2sjy2 理想化的关键因子:
  漂移不动点方向 d* 由内层残差决定, 残差含三类概率项:
    (a) prob1/prob2 的 σ 尺度  (高斯概率场宽度)
    (b) probTotal 的宽度 (|d1+d2-2a| 的约束强度, 乘 width_mult)
    (c) 自洽项权重 w_self
  扫描每个因子, 看 d* 与 真实轴 u (椭球已知) 的夹角能否变小 → 理想化因子
"""
import math
import time

import numpy as np

import data_driven_axis as dd
import n2sjy2 as n2


def angle(u, v):
    return math.degrees(math.acos(np.clip(np.dot(u, v), -1, 1)))


def ellip_axis(seed):
    rng = np.random.default_rng(300 + seed)
    u = rng.standard_normal(3)
    return u / np.linalg.norm(u)


def setup(pts):
    rp = n2.compute_rp(pts)
    d2 = math.asin(math.cos(math.radians(30)) / math.pi)
    a = rp * (1 + math.sin(d2))
    _, r45, r30, _ = n2.yzqx(pts, rp)
    t1t, t2t = n2.compute_taus()
    pt0, p10, p20, sig0 = dd.fast_probs(r30, r45, t1t, t2t, a)
    F10, F20 = n2.extract_foci(pts, p10, sig0[1], p20, sig0[2])
    d0 = np.array(F10, float) - np.array(F20, float)
    d0 = d0 / np.linalg.norm(d0)
    v = n2.pca_axis(pts)
    return r30, r45, a, v, d0


def drift_params(pts, d0, sig_scale=1.0, width_mult=1.0, w_self=1.0,
                 max_outer=40, n_inner=8, omega=0.9, tol_deg=0.5):
    """参数化漂移: 修改 σ 尺度 / probTotal 宽度 / 自洽权重, 返回不动点方向 d*"""
    r30, r45, a, v, _ = setup(pts)
    Z1 = np.asarray(r30, complex)
    Z2 = np.asarray(r45, complex)
    P = np.asarray(pts, float)
    mg = np.arange(-20, 21)
    MM, NN = np.meshgrid(mg, mg)
    MC, NC = MM.ravel(), NN.ravel()

    t1 = t2 = complex(0, 1)
    d = d0.copy()

    def lm_step(x, axis, lam):
        t1c = complex(x[0], x[1])
        t2c = complex(x[2], x[3])
        L1 = MC + NC * t1c
        L2 = MC + NC * t2c
        D1 = np.abs(Z1[:, None] - L1[None, :])
        d1 = D1.min(axis=1)
        D2 = np.abs(Z2[:, None] - L2[None, :])
        d2 = D2.min(axis=1)
        s1 = sig_scale * np.sqrt(np.mean(d1 ** 2))
        s2 = sig_scale * np.sqrt(np.mean(d2 ** 2))
        p1 = np.exp(-d1 ** 2 / (2 * s1 ** 2))
        p2 = np.exp(-d2 ** 2 / (2 * s2 ** 2))
        pt = np.exp(-np.abs(d1 + d2 - 2 * a) / (2 * a * width_mult))
        p0 = P[0]
        A = 2.0 * (P[1:] - p0)
        b = (P[1:] ** 2).sum(1) - (p0 ** 2).sum() - (d1[1:] ** 2 - d1[0] ** 2)
        F1, *_ = np.linalg.lstsq(A, b, rcond=None)
        b2 = (P[1:] ** 2).sum(1) - (p0 ** 2).sum() - (d2[1:] ** 2 - d2[0] ** 2)
        F2, *_ = np.linalg.lstsq(A, b2, rcond=None)
        pf = np.array([np.exp(-abs(np.linalg.norm(p - F1) + np.linalg.norm(p - F2) - 2 * a) / (2 * a))
                       for p in P])
        n = len(P)
        r = np.zeros(n + 9)
        inv = 1.0 / math.sqrt(n)
        r[:n] = w_self * (pt - pf) * inv
        dv = F1 - F2
        if np.linalg.norm(dv) < 1e-12:
            dv = axis
        dv = dv / np.linalg.norm(dv)
        if np.dot(dv, axis) < 0:
            dv = -dv
        e = dv - axis
        r[n:n + 3] = 20.0 * e
        r[n + 7] = 1.0 * (x[1] - 1.0)
        r[n + 8] = 1.0 * (x[3] - 1.0)
        cost = float(np.sum(r ** 2))
        J = np.zeros((n + 9, 4))
        for k in range(4):
            xp = x.copy()
            xp[k] += 1e-3
            t1p = complex(xp[0], xp[1])
            t2p = complex(xp[2], xp[3])
            L1p = MC + NC * t1p
            L2p = MC + NC * t2p
            d1p = np.abs(Z1[:, None] - L1p[None, :]).min(axis=1)
            d2p = np.abs(Z2[:, None] - L2p[None, :]).min(axis=1)
            s1p = sig_scale * np.sqrt(np.mean(d1p ** 2))
            s2p = sig_scale * np.sqrt(np.mean(d2p ** 2))
            p1p = np.exp(-d1p ** 2 / (2 * s1p ** 2))
            p2p = np.exp(-d2p ** 2 / (2 * s2p ** 2))
            ptp = np.exp(-np.abs(d1p + d2p - 2 * a) / (2 * a * width_mult))
            bp = (P[1:] ** 2).sum(1) - (p0 ** 2).sum() - (d1p[1:] ** 2 - d1p[0] ** 2)
            F1p, *_ = np.linalg.lstsq(A, bp, rcond=None)
            b2p = (P[1:] ** 2).sum(1) - (p0 ** 2).sum() - (d2p[1:] ** 2 - d2p[0] ** 2)
            F2p, *_ = np.linalg.lstsq(A, b2p, rcond=None)
            pfp = np.array([np.exp(-abs(np.linalg.norm(p - F1p) + np.linalg.norm(p - F2p) - 2 * a) / (2 * a))
                            for p in P])
            rp = np.zeros(n + 9)
            rp[:n] = w_self * (ptp - pfp) * inv
            dvp = F1p - F2p
            if np.linalg.norm(dvp) < 1e-12:
                dvp = axis
            dvp = dvp / np.linalg.norm(dvp)
            if np.dot(dvp, axis) < 0:
                dvp = -dvp
            ep = dvp - axis
            rp[n:n + 3] = 20.0 * ep
            rp[n + 7] = 1.0 * (xp[1] - 1.0)
            rp[n + 8] = 1.0 * (xp[3] - 1.0)
            J[:, k] = (rp - r) / 1e-3
        A_ = J.T @ J
        g = J.T @ r
        Aaug = A_ + lam * np.diag(np.diag(A_) + 1e-12)
        try:
            delta = np.linalg.solve(Aaug, g)
        except np.linalg.LinAlgError:
            return x, lam
        xtry = x - delta
        t1q = complex(xtry[0], xtry[1])
        t2q = complex(xtry[2], xtry[3])
        L1q = MC + NC * t1q
        L2q = MC + NC * t2q
        d1q = np.abs(Z1[:, None] - L1q[None, :]).min(axis=1)
        d2q = np.abs(Z2[:, None] - L2q[None, :]).min(axis=1)
        s1q = sig_scale * np.sqrt(np.mean(d1q ** 2))
        s2q = sig_scale * np.sqrt(np.mean(d2q ** 2))
        ptq = np.exp(-np.abs(d1q + d2q - 2 * a) / (2 * a * width_mult))
        p1q = np.exp(-d1q ** 2 / (2 * s1q ** 2))
        p2q = np.exp(-d2q ** 2 / (2 * s2q ** 2))
        bq = (P[1:] ** 2).sum(1) - (p0 ** 2).sum() - (d1q[1:] ** 2 - d1q[0] ** 2)
        F1q, *_ = np.linalg.lstsq(A, bq, rcond=None)
        b2q = (P[1:] ** 2).sum(1) - (p0 ** 2).sum() - (d2q[1:] ** 2 - d2q[0] ** 2)
        F2q, *_ = np.linalg.lstsq(A, b2q, rcond=None)
        pfq = np.array([np.exp(-abs(np.linalg.norm(p - F1q) + np.linalg.norm(p - F2q) - 2 * a) / (2 * a))
                        for p in P])
        rq = np.zeros(n + 9)
        rq[:n] = w_self * (ptq - pfq) * inv
        dvq = F1q - F2q
        if np.linalg.norm(dvq) < 1e-12:
            dvq = axis
        dvq = dvq / np.linalg.norm(dvq)
        if np.dot(dvq, axis) < 0:
            dvq = -dvq
        eq = dvq - axis
        rq[n:n + 3] = 20.0 * eq
        rq[n + 7] = 1.0 * (xtry[1] - 1.0)
        rq[n + 8] = 1.0 * (xtry[3] - 1.0)
        cost2 = float(np.sum(rq ** 2))
        if cost2 < cost:
            return xtry, max(lam / 3.0, 1e-8)
        return x, min(lam * 3.0, 1e4)

    for k in range(max_outer):
        x = np.array([t1.real, t1.imag, t2.real, t2.imag])
        lam = 1e-3
        for _ in range(n_inner):
            x, lam = lm_step(x, d, lam)
        t1n, t2n = complex(x[0], x[1]), complex(x[2], x[3])
        L1 = MC + NC * t1n
        L2 = MC + NC * t2n
        d1n = np.abs(Z1[:, None] - L1[None, :]).min(axis=1)
        d2n = np.abs(Z2[:, None] - L2[None, :]).min(axis=1)
        p0 = P[0]
        A = 2.0 * (P[1:] - p0)
        b = (P[1:] ** 2).sum(1) - (p0 ** 2).sum() - (d1n[1:] ** 2 - d1n[0] ** 2)
        F1n, *_ = np.linalg.lstsq(A, b, rcond=None)
        b2 = (P[1:] ** 2).sum(1) - (p0 ** 2).sum() - (d2n[1:] ** 2 - d2n[0] ** 2)
        F2n, *_ = np.linalg.lstsq(A, b2, rcond=None)
        d_new = F1n - F2n
        if np.linalg.norm(d_new) < 1e-12:
            break
        d_new = d_new / np.linalg.norm(d_new)
        chg = angle(d_new, d)
        d = d + omega * (d_new - d)
        d = d / np.linalg.norm(d)
        t1, t2 = t1n, t2n
        if chg < tol_deg:
            break
    return d


def main():
    for seed in (0, 1):
        pts = dd.make_ellip(seed)
        u = ellip_axis(seed)
        r30, r45, a, v, d0 = setup(pts)
        print(f"=== ellip{seed}  真实轴 u={u.round(3)}  ∠(d0,u)={angle(d0,u):.1f}°  "
              f"∠(d0,v)={angle(d0,v):.1f}°  ∠(u,v)={angle(u,v):.1f}° ===", flush=True)
        base = drift_params(pts, d0)
        print(f"  基线: ∠(d*,u)={angle(base, u):6.1f}°  ∠(d*,v)={angle(base, v):6.1f}°", flush=True)
        print("  σ尺度:", flush=True)
        for sc in (0.5, 2.0, 3.0):
            d_ = drift_params(pts, d0, sig_scale=sc)
            print(f"    σ×{sc:<4}: ∠(d*,u)={angle(d_, u):6.1f}°  ∠(d*,v)={angle(d_, v):6.1f}°", flush=True)
        print("  probTotal宽度:", flush=True)
        for wm in (0.5, 2.0, 4.0):
            d_ = drift_params(pts, d0, width_mult=wm)
            print(f"    宽×{wm:<4}: ∠(d*,u)={angle(d_, u):6.1f}°  ∠(d*,v)={angle(d_, v):6.1f}°", flush=True)
        print("  自洽权重:", flush=True)
        for ws in (0.3, 3.0, 10.0):
            d_ = drift_params(pts, d0, w_self=ws)
            print(f"    wSelf={ws:<4}: ∠(d*,u)={angle(d_, u):6.1f}°  ∠(d*,v)={angle(d_, v):6.1f}°", flush=True)
        print(flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
