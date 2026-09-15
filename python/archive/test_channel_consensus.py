# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""不同数据通道的共识是否与PCA相关: 通道 A=(r30,r45), B=(r74,r45), C=(r74,r30)"""
import math

import numpy as np

import data_driven_axis as dd
import n2sjy2 as n2


def angle(u, v):
    return math.degrees(math.acos(np.clip(np.dot(u, v), -1, 1)))


def channel_direction(pts, z1, z2):
    """由通道 (z1格, z2格) 在理论τ处的焦点方向"""
    rp = n2.compute_rp(pts)
    d2 = math.asin(math.cos(math.radians(30)) / math.pi)
    a = rp * (1 + math.sin(d2))
    t1t, t2t = n2.compute_taus()
    pt, p1, p2, sig = dd.fast_probs(z1, z2, t1t, t2t, a)
    F1, F2 = n2.extract_foci(pts, p1, sig[1], p2, sig[2])
    d = np.array(F1, float) - np.array(F2, float)
    return d / np.linalg.norm(d), a


def main():
    datasets = {"cube0": dd.make_cube(10), "cube1": dd.make_cube(11),
                "ball0": dd.make_ball(0), "ball1": dd.make_ball(1),
                "ellip0": dd.make_ellip(0), "ellip1": dd.make_ellip(1)}
    print("=" * 100)
    print("通道共识 ∠(normalize(dX+dY), v_pca): A=(r30,r45) B=(r74,r45) C=(r74,r30)")
    print("=" * 100)
    print(f"{'name':8s} {'∠(A,v)':>7s} {'∠(B,v)':>7s} {'∠(C,v)':>7s} "
          f"{'∠(A+B)':>7s} {'∠(A+C)':>7s} {'∠(B+C)':>7s} {'最好':>7s}")
    for name, pts in datasets.items():
        rp = n2.compute_rp(pts)
        d2 = math.asin(math.cos(math.radians(30)) / math.pi)
        a = rp * (1 + math.sin(d2))
        _, r45, r30, r74 = n2.yzqx(pts, rp)
        v = n2.pca_axis(pts)
        dA, _ = channel_direction(pts, r30, r45)
        dB, _ = channel_direction(pts, r74, r45)
        dC, _ = channel_direction(pts, r74, r30)
        angs = {}
        for k, dk in (("A", dA), ("B", dB), ("C", dC)):
            angs[k] = angle(dk, v)
        cons = {}
        for k1, k2 in (("A", "B"), ("A", "C"), ("B", "C")):
            dc = locals()[f"d{k1}"] + locals()[f"d{k2}"]
            dc = dc / np.linalg.norm(dc)
            cons[f"{k1}+{k2}"] = angle(dc, v)
        best = min(cons.values())
        print(f"{name:8s} {angs['A']:7.1f} {angs['B']:7.1f} {angs['C']:7.1f} "
              f"{cons['A+B']:7.1f} {cons['A+C']:7.1f} {cons['B+C']:7.1f} {best:7.1f}")
    print("DONE")


if __name__ == "__main__":
    main()
