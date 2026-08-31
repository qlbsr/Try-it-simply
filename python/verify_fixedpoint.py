# -*- coding: utf-8 -*-
"""
用户分层判据验证:
  档1: PCA 环收敛 (θ_final<0.5°) → 成立
  档2: PCA 环未收敛 → 跑无 v3 概率修正环 500 次,
       若角度自收敛到固定值 (后100次迭代范围≤3°) → 成立, 否则失败
同时输出 档2 结束时 t1,t2 的对称性 (|t1-t2|, D1)
"""
import math

import numpy as np

import nsjy_algorithms as m
import verify_probability_correction as vp

t10, t20 = m.compute_taus()


def fit_focus_np(P, dist):
    """等价 nsjy.FitFocus (numpy)"""
    p0 = P[0]
    d0 = dist[0]
    A = 2.0 * (P[1:] - p0)
    b = (P[1:] ** 2).sum(1) - (p0 ** 2).sum() - (dist[1:] ** 2 - d0 ** 2)
    F, *_ = np.linalg.lstsq(A, b, rcond=None)
    return F


def prob_loop(pts, r30, r45, a, t1, t2, n_iter=500, eta=0.3, rng=20, every=25):
    """总概率加权修正环 (无 v3)。返回历史 (it, t1, t2, dn)。全向量化。"""
    Z1 = np.asarray(r30, dtype=complex)
    Z2 = np.asarray(r45, dtype=complex)
    P = np.asarray(pts, dtype=float)
    mg = np.arange(-rng, rng + 1)
    MM, NN = np.meshgrid(mg, mg)
    MC = MM.ravel()
    NC = NN.ravel()
    hist = []
    for it in range(n_iter + 1):
        L1 = MC + NC * t1
        L2 = MC + NC * t2
        D1 = np.abs(Z1[:, None] - L1[None, :])
        j1 = np.argmin(D1, axis=1)
        d1 = D1[np.arange(len(Z1)), j1]
        D2 = np.abs(Z2[:, None] - L2[None, :])
        j2 = np.argmin(D2, axis=1)
        d2 = D2[np.arange(len(Z2)), j2]
        s1 = np.sqrt(np.mean(d1 * d1))
        s2 = np.sqrt(np.mean(d2 * d2))
        p1 = np.exp(-d1 * d1 / (2.0 * s1 * s1))
        p2 = np.exp(-d2 * d2 / (2.0 * s2 * s2))
        pt = np.exp(-np.abs(d1 + d2 - 2.0 * a) / (2.0 * a))
        F1 = fit_focus_np(P, d1)
        F2 = fit_focus_np(P, d2)
        dvec = F1 - F2
        nrm = np.linalg.norm(dvec)
        dn = dvec / nrm if nrm > 1e-9 else None
        w = pt
        W = w.sum() if w.sum() > 1e-9 else 1.0
        g1 = np.sum(w * NC[j1] * (Z1 - L1[j1])) / W
        g2 = np.sum(w * NC[j2] * (Z2 - L2[j2])) / W
        st = abs(g1) + abs(g2)
        cap = 0.05
        if st > cap:
            g1 *= cap / st
            g2 *= cap / st
        if it % every == 0:
            hist.append((it, t1, t2, dn))
        t1 = t1 + eta * g1
        t2 = t2 + eta * g2
    return hist


def main():
    names = ([f"ball{i}" for i in range(5)] + [f"gauss{i}" for i in range(5)]
             + ["ellip0", "ellip1", "ellipN0", "ellipN1"])
    print("=" * 118)
    print("档1: PCA环收敛(<0.5°)  档2: 概率环500次自收敛(后100次角度范围≤3°)")
    print("=" * 118)
    hdr = (f"{'name':8s} {'PCA环θ':>8s} {'档1':>5s} | {'概率环θ0':>7s} {'θ500':>7s} "
           f"{'尾100范围':>9s} {'档2自收敛':>7s} {'固定角':>7s} | {'|t1-t2|':>7s} {'D1':>6s}")
    print(hdr)
    for name in names:
        pts = vp.build(name)
        r30, r45, a, axis = vp.prepare(pts)

        # 档1: PCA 环 (复用批量结果, 没有的现场跑 40 次)
        if name in vp.BATCH:
            th_pca = vp.BATCH[name][2]
        else:
            _, _, F1, F2 = m.refine_moduli_by_axis(pts, t10, t20, axis,
                                                   max_iter=40, verbose=False)
            d = F1 - F2
            dn = d / np.linalg.norm(d)
            th_pca = vp.ang_line(dn, axis)
        tier1 = "成立" if th_pca < 0.5 else "失败"

        # 档2: 概率环 500 次
        hist = prob_loop(pts, r30, r45, a, t10, t20, n_iter=500)
        angs = [vp.ang_line(dn, axis) if dn is not None else float("nan")
                for _, _, _, dn in hist]
        tail = angs[-4:]                      # 最后 100 次迭代 (每25取1)
        spread = max(tail) - min(tail)
        settled = spread <= 3.0
        tier2 = "成立" if settled else "失败"
        fixed = (sum(tail) / len(tail)) if settled else float("nan")
        _, tf1, tf2, _ = hist[-1]
        d12 = abs(tf1 - tf2)
        D1 = abs(tf1 - 1j) + abs(tf2 - 1j)
        print(f"{name:8s} {th_pca:8.2f} {tier1:>5s} | {angs[0]:7.1f} {angs[-1]:7.1f} "
              f"{min(tail):8.1f}-{max(tail):<4.1f} {tier2:>7s} "
              f"{fixed:6.1f}° | {d12:7.3f} {D1:6.3f}  t1f=({tf1.real:.3f},{tf1.imag:.3f}) "
              f"t2f=({tf2.real:.3f},{tf2.imag:.3f})")

    print()
    print("=== ball0 完整轨迹 (每25次) ===")
    pts = vp.build("ball0")
    r30, r45, a, axis = vp.prepare(pts)
    hist = prob_loop(pts, r30, r45, a, t10, t20, n_iter=500)
    for it, t1, t2, dn in hist:
        if dn is None:
            continue
        print(f"it={it:3d}  θ={vp.ang_line(dn, axis):6.1f}°  "
              f"t1=({t1.real:.3f},{t1.imag:.3f}) t2=({t2.real:.3f},{t2.imag:.3f})")
    print("DONE")


if __name__ == "__main__":
    main()
