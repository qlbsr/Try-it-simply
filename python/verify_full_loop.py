# -*- coding: utf-8 -*-
"""
忠实复刻 n2sjy2.cs 修正版 整个迭代 (不拆散):
  每轮: 内层LM(轴=当前焦点方向) → 重拟合 → angleDeg(与v)/angleDeg1(与初始F1F2)
        → 旋转 v 和 (F1-F2) 到当前方向 → NM(放大趋势) → 更新 t1,t2
  跟踪: 每轮 angle(d,v) 是否随迭代 →≤10 (NM 放大摆动/趋势)
"""
import math
import time

import numpy as np
import scipy.optimize as opt

import data_driven_axis as dd
import n2sjy2 as n2


def angle(u, v):
    return math.degrees(math.acos(np.clip(np.dot(u, v), -1, 1)))


def setup(pts):
    rp = n2.compute_rp(pts)
    d2 = math.asin(math.cos(math.radians(30)) / math.pi)
    a = rp * (1 + math.sin(d2))
    _, r45, r30, _ = n2.yzqx(pts, rp)
    v = n2.pca_axis(pts)
    t1t, t2t = n2.compute_taus()
    pt0, p10, p20, sig0 = dd.fast_probs(r30, r45, t1t, t2t, a)
    F10, F20 = n2.extract_foci(pts, p10, sig0[1], p20, sig0[2])
    d0 = np.array(F10, float) - np.array(F20, float)
    d0 = d0 / np.linalg.norm(d0)
    return r30, r45, a, v, d0, t1t, t2t


def rotate_vec(ref, axis, ang_deg):
    """绕 axis 旋转 ang_deg (Rodrigues) — 与 Quaternion.AngleAxis 等价"""
    ax = axis / np.linalg.norm(axis)
    th = math.radians(ang_deg)
    return (ref * math.cos(th) + np.cross(ax, ref) * math.sin(th)
            + ax * np.dot(ax, ref) * (1 - math.cos(th)))


def nm_amplify(pts, r30, r45, a, v_new, F_new, x0, max_evals=400):
    """NM: 最小化 |angle(d,newv) − angle(d,newF)| (用户 RefineTausWithNM)"""
    evals = [0]

    def cost(x):
        evals[0] += 1
        t1 = complex(x[0], x[1])
        t2 = complex(x[2], x[3])
        pt, p1, p2, sig = dd.fast_probs(r30, r45, t1, t2, a)
        F1, F2 = n2.extract_foci(pts, p1, sig[1], p2, sig[2])
        d = np.array(F1, float) - np.array(F2, float)
        d = d / np.linalg.norm(d)
        return abs(angle(d, v_new) - angle(d, F_new))

    def cb(xk):
        return cost(xk) <= 10.0

    x0c = np.array([max(-0.5, min(0.5, x0[0])), max(0.5, min(1.5, x0[1])),
                    max(-0.5, min(0.5, x0[2])), max(0.5, min(1.5, x0[3]))])
    bounds = [(-0.5, 0.5), (0.5, 1.5), (-0.5, 0.5), (0.5, 1.5)]
    res = opt.minimize(cost, x0c, method="Nelder-Mead", bounds=bounds, callback=cb,
                       options={"maxiter": max_evals, "maxfev": max_evals * 2})
    return res.x, evals[0]


def full_loop(pts, rounds=40, inner=15):
    r30, r45, a, v, d0, t1t, t2t = setup(pts)
    t1, t2 = t1t, t2t                     # 初始=理论
    # 初始 f1,f2 (理论τ焦点)
    pt0, p10, p20, sig0 = dd.fast_probs(r30, r45, t1, t2, a)
    F1, F2 = n2.extract_foci(pts, p10, sig0[1], p20, sig0[2])
    f1, f2 = np.array(F1, float), np.array(F2, float)
    # 初始 vj = normalize(v + (F1-F2)) — 用户 line 43
    vj = v + (f1 - f2) / np.linalg.norm(f1 - f2)
    vj = vj / np.linalg.norm(vj)
    # 初始 RefineModuliByAxis(axis=vj) — 用户 line 44
    t1n, t2n, F1n, F2n = dd.inner_refine(complex(0, 1), complex(0, 1), pts, r30, r45, a, vj, inner)
    t1, t2 = t1n, t2n

    hist = []
    for i in range(rounds):
        pt, p1, p2, sig = dd.fast_probs(r30, r45, t1, t2, a)
        # 内层LM: 轴=当前焦点方向 (用户 line 48)
        axis_now = (f1 - f2)
        axis_now = axis_now / np.linalg.norm(axis_now)
        t1b, t2b, F1b, F2b = dd.inner_refine(complex(0, 1), complex(0, 1), pts, r30, r45, a, axis_now, inner)
        # 重拟合 (用户 FitFociByProbability)
        ptb, p1b, p2b, sigb = dd.fast_probs(r30, r45, t1b, t2b, a)
        F1f, F2f = n2.extract_foci(pts, p1b, sigb[1], p2b, sigb[2])
        d_new = np.array(F1f, float) - np.array(F2f, float)
        d_new = d_new / np.linalg.norm(d_new)
        angleDeg = angle(d_new, v)
        angleDeg1 = angle(d_new, d0)
        hist.append((i, angleDeg, angleDeg1, (t1b, t2b)))
        if (angleDeg + angleDeg1) * 0.5 - min(angleDeg, angleDeg1) <= 5:
            break
        # 旋转参考 (用户 line 54-60)
        axis_c = np.cross(d_new, v)
        axis_c1 = np.cross(d_new, d0)
        newv = rotate_vec(v, axis_c, -angleDeg)
        newF = rotate_vec(d0, axis_c1, -angleDeg1)
        # NM 放大 (用户 RefineTausWithNM)
        x0 = [t1b.real, t1b.imag, t2b.real, t2b.imag]
        xf, ev = nm_amplify(pts, r30, r45, a, newv, newF, x0)
        t1, t2 = complex(xf[0], xf[1]), complex(xf[2], xf[3])
        ptf, p1f, p2f, sigf = dd.fast_probs(r30, r45, t1, t2, a)
        F1z, F2z = n2.extract_foci(pts, p1f, sigf[1], p2f, sigf[2])
        f1, f2 = np.array(F1z, float), np.array(F2z, float)
    return hist


def main():
    datasets = {"cube0": dd.make_cube(10), "ball0": dd.make_ball(0),
                "ellip0": dd.make_ellip(0), "ellip1": dd.make_ellip(1)}
    print("=" * 100)
    print("n2sjy2 整个迭代 (内层LM + 旋转 + NM放大): 每轮 angle(d,v) 是否 →≤10")
    print("=" * 100)
    for name, pts in datasets.items():
        t0 = time.time()
        hist = full_loop(pts, rounds=40)
        angs = [h[1] for h in hist]
        deltas = [abs(h[1] - h[2]) for h in hist]
        idx = sorted(set([0, len(angs)//4, len(angs)//2, 3*len(angs)//4, len(angs)-1]))
        traj = " → ".join(f"{angs[i]:.0f}" for i in idx if i < len(angs))
        ok = "✓" if angs[-1] <= 10 else "✗"
        print(f"{name:8s} 轮数={len(hist):2d}  angle(d,v): {traj}  末值={angs[-1]:.1f}° ({ok})  "
              f"末|Δ|={deltas[-1]:.1f}°  min|Δ|={min(deltas):.1f}°  [{time.time()-t0:.0f}s]", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
