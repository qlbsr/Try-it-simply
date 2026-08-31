# -*- coding: utf-8 -*-
"""复现自迭代日志: 每轮输出 (Re t1, Im t1, Re t2, Im t2) 与 maxDev, 并做关系分析"""
import numpy as np

import nsjy_algorithms as m
import verify_probability_correction as vp

t10, t20 = m.compute_taus()


def run(pts, r30, r45, a, t1, t2, n_iter=200):
    Z1 = np.asarray(r30, complex)
    Z2 = np.asarray(r45, complex)
    P = np.asarray(pts, float)
    mg = np.arange(-20, 21)
    MM, NN = np.meshgrid(mg, mg)
    MC, NC = MM.ravel(), NN.ravel()
    taus = []
    devs = []
    switch1 = [0]
    last_idx = None
    for it in range(n_iter + 1):
        taus.append((t1.real, t1.imag, t2.real, t2.imag))
        L1 = MC + NC * t1
        L2 = MC + NC * t2
        D1 = np.abs(Z1[:, None] - L1[None, :])
        j1 = np.argmin(D1, axis=1)
        d1 = D1[np.arange(len(Z1)), j1]
        D2 = np.abs(Z2[:, None] - L2[None, :])
        j2 = np.argmin(D2, axis=1)
        d2 = D2[np.arange(len(Z2)), j2]
        # 最近格点切换计数 (格1)
        if last_idx is not None:
            switch1.append(int(np.sum(j1 != last_idx)))
        last_idx = j1.copy()
        s1 = np.sqrt(np.mean(d1 * d1))
        s2 = np.sqrt(np.mean(d2 * d2))
        p1 = np.exp(-d1 * d1 / (2 * s1 * s1))
        p2 = np.exp(-d2 * d2 / (2 * s2 * s2))
        pt = np.exp(-np.abs(d1 + d2 - 2 * a) / (2 * a))
        p0 = P[0]
        A = 2.0 * (P[1:] - p0)
        b = (P[1:] ** 2).sum(1) - (p0 ** 2).sum() - (d1[1:] ** 2 - d1[0] ** 2)
        F1, *_ = np.linalg.lstsq(A, b, rcond=None)
        b2 = (P[1:] ** 2).sum(1) - (p0 ** 2).sum() - (d2[1:] ** 2 - d2[0] ** 2)
        F2, *_ = np.linalg.lstsq(A, b2, rcond=None)  # A 只依赖点坐标, F1/F2 共用
        pf = np.array([np.exp(-abs(np.linalg.norm(p - F1) + np.linalg.norm(p - F2) - 2 * a) / (2 * a))
                       for p in P])
        devs.append(float(np.max(np.abs(pt - pf))))
        if it == n_iter:
            break
        w = pt
        W = w.sum() if w.sum() > 1e-9 else 1.0
        g1 = np.sum(w * NC[j1] * (Z1 - L1[j1])) / W
        g2 = np.sum(w * NC[j2] * (Z2 - L2[j2])) / W
        st = abs(g1) + abs(g2)
        cap = 0.05
        if st > cap:
            g1 *= cap / st
            g2 *= cap / st
        t1 = t1 + 0.3 * g1
        t2 = t2 + 0.3 * g2
    return np.array(taus), np.array(devs), np.array(switch1)


def main():
    pts = vp.build("ball4")
    r30, r45, a, axis = vp.prepare(pts)
    taus, devs, switch1 = run(pts, r30, r45, a, t10, t20, n_iter=200)

    print("=== τ 轨迹前 30 轮 (Re t1, Im t1, Re t2, Im t2) ===")
    for k in range(30):
        print(f"({taus[k,0]:.6f}, {taus[k,1]:.6f}, {taus[k,2]:.6f}, {taus[k,3]:.6f})")
    print("...")

    # 整数度: τ 是否被推向高斯整数
    gint = np.maximum.reduce([
        np.minimum(np.abs(taus[:, 0] - np.round(taus[:, 0])),
                   np.abs(taus[:, 0] - 0)),
        np.minimum(np.abs(taus[:, 1] - np.round(taus[:, 1])),
                   np.abs(taus[:, 1] - 0)),
    ])
    print(f"\nτ1 到最近高斯整数的平均距离: {gint.mean():.4f}  (0=整数)")
    gint2 = np.maximum.reduce([
        np.minimum(np.abs(taus[:, 2] - np.round(taus[:, 2])), np.abs(taus[:, 2] - 0)),
        np.minimum(np.abs(taus[:, 3] - np.round(taus[:, 3])), np.abs(taus[:, 3] - 0)),
    ])
    print(f"τ2 到最近高斯整数的平均距离: {gint2.mean():.4f}")

    # maxDev 统计与周期
    print(f"\nmaxDev: min={devs.min():.3f} max={devs.max():.3f} mean={devs.mean():.3f}")
    # 自相关找周期
    d = devs - devs.mean()
    ac = np.correlate(d, d, "full")[len(d) - 1:]
    ac = ac / ac[0]
    peaks = np.where((ac[1:-1] > ac[:-2]) & (ac[1:-1] > ac[2:]) & (ac[1:-1] > 0.3))[0] + 1
    if len(peaks) >= 2:
        period = int(np.median(np.diff(peaks[:4])))
        print(f"maxDev 自相关周期 ≈ {period} 轮")
    else:
        print("maxDev 无明显周期")

    # 最近格点切换与 maxDev 变化的关系
    dd = np.abs(np.diff(devs))
    switch = switch1[1:]
    corr_ev = float(np.corrcoef(dd, switch.astype(float))[0, 1]) if switch.std() > 0 else float("nan")
    print(f"\n每轮 |ΔmaxDev| 与 格1切换点数 的相关系数: {corr_ev:+.3f}")
    print(f"切换发生的轮次占比: {(switch > 0).mean()*100:.1f}%")

    # 整数度与 maxDev 的关系
    gi_all = (gint + gint2) / 2
    print(f"τ-整数度 与 maxDev 的相关系数: {float(np.corrcoef(gi_all, devs)[0,1]):+.3f}")
    # 分段: 高整数度(τ近整数) 时 maxDev 均值 vs 低整数度
    hi = gi_all > np.median(gi_all)
    print(f"τ 近整数时  maxDev 均值: {devs[hi].mean():.3f}")
    print(f"τ 远离整数时 maxDev 均值: {devs[~hi].mean():.3f}")


if __name__ == "__main__":
    main()
