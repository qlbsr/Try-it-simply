# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
新设想实验: 外环每轮末尾插入 nsjy4 收尾
  轮 k: (a) 内层LM收敛(轴=当前d, 热启动) → (b) 比对 |Δ| → (c) nsjy4 从当前τ优化 |Δ|≤10
        → (d) 新 (t1,t2) 作为下一轮输入, 重复
观察: (t1,t2)/方向 是否收敛(不动点) 或 振荡(极限环), |Δ| 轨迹
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
    return r30, r45, a, v, d0


def angles_at(x, pts, r30, r45, a, v, d0):
    t1 = complex(x[0], x[1])
    t2 = complex(x[2], x[3])
    pt, p1, p2, sig = dd.fast_probs(r30, r45, t1, t2, a)
    F1, F2 = n2.extract_foci(pts, p1, sig[1], p2, sig[2])
    d = np.array(F1, float) - np.array(F2, float)
    d = d / np.linalg.norm(d)
    return (math.degrees(math.acos(np.clip(np.dot(d, v), -1, 1))),
            math.degrees(math.acos(np.clip(np.dot(d, d0), -1, 1))))


def direction_at(x, pts, r30, r45, a):
    t1 = complex(x[0], x[1])
    t2 = complex(x[2], x[3])
    pt, p1, p2, sig = dd.fast_probs(r30, r45, t1, t2, a)
    F1, F2 = n2.extract_foci(pts, p1, sig[1], p2, sig[2])
    d = np.array(F1, float) - np.array(F2, float)
    return d / np.linalg.norm(d)


def nsjy4_refine(pts, r30, r45, a, v, d0, x0, max_evals=400):
    """nsjy4: Nelder-Mead 最小化 |Δ|, 从 x0 热启动"""
    evals = [0]

    def cost(x):
        evals[0] += 1
        ad, ad1 = angles_at(x, pts, r30, r45, a, v, d0)
        return abs(ad - ad1)

    def cb(xk):
        return cost(xk) <= 10.0

    x0c = np.array([max(-0.5, min(0.5, x0[0])), max(0.5, min(3.0, x0[1])),
                    max(-0.5, min(0.5, x0[2])), max(0.5, min(3.0, x0[3]))])
    bounds = [(-0.5, 0.5), (0.5, 3.0), (-0.5, 0.5), (0.5, 3.0)]
    res = opt.minimize(cost, x0c, method="Nelder-Mead", bounds=bounds, callback=cb,
                       options={"maxiter": max_evals, "maxfev": max_evals * 2})
    return res.x, evals[0]


def interleaved(pts, rounds=6, n_inner=15, omega=0.9):
    """外环每轮末尾插入 nsjy4"""
    r30, r45, a, v, d0 = setup(pts)
    t1 = t2 = complex(0, 1)
    d = d0.copy()
    hist = []
    for k in range(rounds):
        # (a) 内层 LM 收敛 (轴=当前 d, 热启动)
        t1b, t2b, F1, F2 = dd.inner_refine(t1, t2, pts, r30, r45, a, d, n_inner)
        # 自由重拟合 → 方向
        pt, p1, p2, sig = dd.fast_probs(r30, r45, t1b, t2b, a)
        F1f, F2f = n2.extract_foci(pts, p1, sig[1], p2, sig[2])
        d_new = np.array(F1f, float) - np.array(F2f, float)
        d_new = d_new / np.linalg.norm(d_new)
        xb = [t1b.real, t1b.imag, t2b.real, t2b.imag]
        delta_b = abs(angles_at(xb, pts, r30, r45, a, v, d0)[0]
                      - angles_at(xb, pts, r30, r45, a, v, d0)[1])
        # (c) nsjy4 收尾
        if delta_b > 10.0:
            xc, ev = nsjy4_refine(pts, r30, r45, a, v, d0, xb)
            t1c, t2c = complex(xc[0], xc[1]), complex(xc[2], xc[3])
            dc = direction_at(xc, pts, r30, r45, a)
            ad, ad1 = angles_at(xc, pts, r30, r45, a, v, d0)
            delta_c = abs(ad - ad1)
            used_nsjy4 = True
        else:
            t1c, t2c = t1b, t2b
            dc = d_new
            delta_c = delta_b
            used_nsjy4 = False
        # 方向作为下一轮轴 (阻尼)
        d = d + omega * (dc - d)
        d = d / np.linalg.norm(d)
        t1, t2 = t1c, t2c
        chg = angle(dc, d_new) if used_nsjy4 else 0.0
        hist.append((k, delta_b, delta_c, angle(dc, v), used_nsjy4,
                     (t1.real, t1.imag, t2.real, t2.imag)))
    return hist


def main():
    datasets = {"cube0": dd.make_cube(10), "ball0": dd.make_ball(0),
                "ellip0": dd.make_ellip(0)}
    for name, pts in datasets.items():
        print(f"=== {name} 外环+nsjy4收尾 (每轮: 内层15步 → 比对 → nsjy4) ===", flush=True)
        t0 = time.time()
        hist = interleaved(pts, rounds=6)
        for k, db, dc, adv, used, tup in hist:
            print(f"  轮{k}: 内层后|Δ|={db:6.2f}° → nsjy4后|Δ|={dc:6.2f}°  "
                  f"∠(d,v)={adv:6.1f}°  {'nsjy4用' if used else '不用'}"
                  f"  τ=({tup[0]:+.3f},{tup[1]:.3f},{tup[2]:+.3f},{tup[3]:.3f})", flush=True)
        print(f"  [{time.time()-t0:.0f}s]", flush=True)
        print(flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
