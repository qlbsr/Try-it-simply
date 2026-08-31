# -*- coding: utf-8 -*-
"""
聚焦: 内层 30 是否不够? 外层冷启动(每次从 (i,i) 出发, 与 n2sjy2 一致), 只改内层
{30, 50, 100}, 对比 最终 angle1 / angle_pca / |Δ| 是否变化.
(保持其他一切相同, 自由重拟合用 extract_foci 以控制变量)
"""
import math

import numpy as np

import data_driven_axis as dd
import n2sjy2 as n2


def outer_cold(points, d0, n_inner, outer=25):
    rp = n2.compute_rp(points)
    d2 = math.asin(math.cos(math.radians(30)) / math.pi)
    a = rp * (1 + math.sin(d2))
    _, r45, r30, _ = n2.yzqx(points, rp)
    v_pca = n2.pca_axis(points)
    init_dir = np.array(d0, float)
    init_dir = init_dir / np.linalg.norm(init_dir)

    d = init_dir
    rec = []
    for k in range(outer):
        # 内层: 冷启动 (i,i), 方向目标 = 当前 d
        t1n, t2n, F1, F2 = dd.inner_refine(complex(0, 1), complex(0, 1),
                                           points, r30, r45, a, d, n_inner)
        pt, p1, p2, sig = dd.fast_probs(r30, r45, t1n, t2n, a)
        F1f, F2f = n2.extract_foci(points, p1, sig[1], p2, sig[2])
        d_new = np.array(F1f, float) - np.array(F2f, float)
        if np.linalg.norm(d_new) < 1e-12:
            break
        d_new = d_new / np.linalg.norm(d_new)
        angle1 = math.degrees(math.acos(np.clip(abs(np.dot(d_new, init_dir)), -1, 1)))
        angle_pca = math.degrees(math.acos(np.clip(abs(np.dot(d_new, v_pca)), -1, 1)))
        d = d_new                                  # 无阻尼, 与原 n2sjy2 一致
        rec.append((angle1, angle_pca))
    return rec


def main():
    datasets = {"cube0": dd.make_cube(10), "ellip0": dd.make_ellip(0)}
    print("=" * 96)
    print("内层 {30,50,100} × 外层冷启动25: 最终 angle1 / angle_pca / |Δ| 是否随内层变化")
    print("=" * 96, flush=True)
    for name, pts in datasets.items():
        d0 = dd.init_direction(pts)
        print(f"--- {name} ---", flush=True)
        finals = {}
        for inner in (30, 50, 100):
            rec = outer_cold(pts, d0, inner, outer=25)
            a1, ap = rec[-1]
            a1_0, ap_0 = rec[0]
            dmin = min(abs(r[0] - r[1]) for r in rec)
            finals[inner] = (a1, ap)
            print(f"  内层={inner:3d}: 首轮(angle1={a1_0:5.1f}, angle_pca={ap_0:5.1f}) → "
                  f"末轮(angle1={a1:5.1f}, angle_pca={ap:5.1f})  |Δ|end={abs(a1-ap):5.1f}°  "
                  f"全程min|Δ|={dmin:5.1f}°", flush=True)
        # 一致性: 内层30 vs 50 vs 100 的末轮角差
        a1_30, ap_30 = finals[30]
        a1_50, ap_50 = finals[50]
        a1_100, ap_100 = finals[100]
        print(f"  → 内层30/50/100 末轮 angle_pca 差值: |{ap_30:.1f}-{ap_50:.1f}|={abs(ap_30-ap_50):.1f}°, "
              f"|{ap_50:.1f}-{ap_100:.1f}|={abs(ap_50-ap_100):.1f}°", flush=True)
        print(f"  → 末轮|Δ| 差值: 30={abs(a1_30-ap_30):.1f}°  50={abs(a1_50-ap_50):.1f}°  "
              f"100={abs(a1_100-ap_100):.1f}°", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
