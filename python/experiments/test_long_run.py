# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
聚焦测试: "收敛即停"下未收敛到PCA的案例, 是否因为迭代次数不够?
  长跑外层(紧判据 |Δd|<0.01°, 上限300), 内层=10(热启动) 与 内层=收敛即停 两种
  追踪 angle(d, v_pca) 轨迹 → 若持续下降至0 = 次数不够; 若停在平台 = 结构性失败
"""
import math
import time

import numpy as np

import data_driven_axis as dd
import n2sjy2 as n2


def long_run(points, d0, n_inner, inner_tol, omega=0.9, max_outer=300, tol_deg=0.01):
    rp = n2.compute_rp(points)
    d2 = math.asin(math.cos(math.radians(30)) / math.pi)
    a = rp * (1 + math.sin(d2))
    _, r45, r30, _ = n2.yzqx(points, rp)
    v_pca = n2.pca_axis(points)

    t1 = t2 = complex(0, 1)
    d = np.array(d0, float)
    d = d / np.linalg.norm(d)
    hist = []
    iters = 0
    for k in range(max_outer):
        t1, t2, F1, F2 = dd.inner_refine(t1, t2, points, r30, r45, a, d, n_inner, inner_tol)
        pt, p1, p2, sig = dd.fast_probs(r30, r45, t1, t2, a)
        F1f, F2f = n2.extract_foci(points, p1, sig[1], p2, sig[2])
        d_new = np.array(F1f, float) - np.array(F2f, float)
        if np.linalg.norm(d_new) < 1e-12:
            break
        d_new = d_new / np.linalg.norm(d_new)
        chg = math.degrees(math.acos(np.clip(np.dot(d_new, d), -1, 1)))
        d = d + omega * (d_new - d)
        d = d / np.linalg.norm(d)
        ang = dd.angle(d, v_pca)
        hist.append((k, chg, ang))
        iters = k + 1
        if chg < tol_deg:
            break
    return d, hist, iters


def main():
    datasets = {"cube0": dd.make_cube(10), "cube1": dd.make_cube(11),
                "ball0": dd.make_ball(0), "ellip0": dd.make_ellip(0)}
    for mode, n_inner, inner_tol in (("内层=10", 10, None),
                                     ("内层=收敛即停", 300, 1e-6)):
        print("=" * 100)
        print(f"{mode}: 外层紧判据|Δd|<0.01°, 上限300, 阻尼0.9")
        print("=" * 100)
        for name, pts in datasets.items():
            d0 = dd.init_direction(pts)
            t0 = time.time()
            d, hist, iters = long_run(pts, d0, n_inner, inner_tol)
            angs = [h[2] for h in hist]
            # 轨迹采样: 首/1/4/1/2/3/4/末
            idx = sorted(set([0, len(angs)//4, len(angs)//2, 3*len(angs)//4, len(angs)-1]))
            traj = " → ".join(f"{angs[i]:.1f}" for i in idx if i < len(angs))
            tail = angs[-5:]
            stable = (max(tail) - min(tail)) < 1.0
            print(f"{name:8s} 停止于外层={iters:3d}  ang(d*,v_pca)={angs[-1]:6.2f}°  "
                  f"尾5角范围={max(tail)-min(tail):5.2f}°  "
                  f"{'已平台' if stable else '仍漂移'}")
            print(f"         轨迹(采5点): {traj}")
            print(f"         [{time.time()-t0:.0f}s]")
    print("DONE")


if __name__ == "__main__":
    main()
