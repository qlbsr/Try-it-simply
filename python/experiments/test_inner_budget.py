# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
内层预算完整扫描: 内层 = {5, 15, 30, 50, 100, 200, 收敛即停}
观察: 收敛次数 / angle(d*,v_pca) / d* 是否随内层→∞ 趋于稳定极限
(先不改任何代码, 只把内层条件测完)
"""
import math
import time

import numpy as np

import data_driven_axis as dd
import n2sjy2 as n2

BUDGETS = [5, 15, 30, 50, 100, 200, "conv"]   # conv = 内层迭代至 |Δx|<1e-6 (上限300)


def main():
    datasets = {}
    datasets["cube0"] = dd.make_cube(10)
    datasets["cube1"] = dd.make_cube(11)
    datasets["ball0"] = dd.make_ball(0)
    datasets["ellip0"] = dd.make_ellip(0)

    print("=" * 100)
    print("内层预算扫描: {5,15,30,50,100,200,收敛即停}  外层上限200 阻尼ω=0.9")
    print("=" * 100)
    for name, pts in datasets.items():
        d0 = dd.init_direction(pts)
        v_pca = n2.pca_axis(pts)
        outs = {}
        for b in BUDGETS:
            t0 = time.time()
            if b == "conv":
                d, _, _, _, iters = dd.data_driven_axis(pts, d0, n_inner=300,
                                                        inner_tol=1e-6, max_outer=200)
            else:
                d, _, _, _, iters = dd.data_driven_axis(pts, d0, n_inner=b, max_outer=200)
            ang = dd.angle(d, v_pca)
            outs[b] = (iters, ang, d)
            dt = time.time() - t0
            print(f"{name:8s} 内层={str(b):>4s} 外层={iters:3d}  angle(d*,v_pca)={ang:6.2f}°  "
                  f"({dt:4.0f}s)")
        # d* 随内层预算的稳定性: 相邻预算方向夹角
        prev = None
        line = f"{name:8s} 相邻预算Δd*: "
        for b in BUDGETS:
            if prev is not None:
                line += f"{b}:{dd.angle(outs[prev][2], outs[b][2]):5.2f}°  "
            prev = b
        # 末尾三段(100,200,conv) 的最大夹角 = 极限稳定性
        tail = [outs[b][2] for b in (100, 200, "conv")]
        tailmax = max(dd.angle(tail[0], tail[1]), dd.angle(tail[1], tail[2]),
                      dd.angle(tail[0], tail[2]))
        print(line)
        print(f"         内层100/200/收敛即停 三者最大Δd* = {tailmax:5.2f}°  "
              f"{'极限稳定' if tailmax < 5 else '极限不稳定(仍漂移)'}")
        print("-" * 100)
    print("DONE")


if __name__ == "__main__":
    main()
