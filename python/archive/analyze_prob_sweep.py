# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
用户规律精确验证: 固定数据集, 让方向 v5 从 v1 旋转到 v (扫描夹角),
  记录 P(v5)=BatchProbability(v5·c, v5·-c)[0] 随 ∠(v5,v), ∠(v5,v1) 的变化
  目标: 找到 P(方向) 与夹角的关系 → 用概率替代角度终止条件 (脱离 PCA)
"""
import json
import math
import numpy as np

import data_driven_axis as dd
import n2sjy2 as n2
import nsjy_algorithms as m

PYJSON = r"C:\Users\23128\My project (2)\Assets\Resources\pyjson.json"


def angle(u, v):
    return math.degrees(math.acos(np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1)))


def batch_probability(points, F1, F2, a):
    P = np.asarray(points, float)
    d1 = np.linalg.norm(P - np.asarray(F1, float), axis=1)
    d2 = np.linalg.norm(P - np.asarray(F2, float), axis=1)
    delta = d1 + d2 - 2.0 * a
    return np.exp(-np.abs(delta) / (2.0 * a))


def rotate_vec(v, axis, deg):
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return v.copy()
    axis = axis / n
    ang = math.radians(deg)
    return (v * math.cos(ang) + np.cross(axis, v) * math.sin(ang)
            + axis * np.dot(axis, v) * (1 - math.cos(ang)))


def setup():
    with open(PYJSON, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    pts = [np.array([p["x"], p["y"], p["z"]], float) for p in raw]
    rp = n2.compute_rp(pts)
    d2 = math.asin(math.cos(math.radians(30)) / math.pi)
    a = rp * (1 + math.sin(d2))
    e3 = math.cos(d2) * 2 / (1 + math.sin(d2))
    h = rp * math.cos(d2)
    c = h * e3
    _, r45, r30, _ = n2.yzqx(pts, rp)
    t1t, t2t = n2.compute_taus()
    t_ = complex(0, 1)
    pt, p1, p2, sig = dd.fast_probs(r30, r45, t1t, t2t, a)
    F1, F2 = m.extract_foci(pts, p1, sig[1], p2, sig[2])
    pt01, p101, p201, sig01 = dd.fast_probs(r30, r45, t_, t_, a)
    F01, F02 = m.extract_foci(pts, p101, sig01[1], p201, sig01[2])
    v = m.pca(pts)[1]
    v = v / np.linalg.norm(v)
    v1 = np.array(F1, float) - np.array(F2, float)
    v1 = v1 / np.linalg.norm(v1)
    return np.array(pts, float), a, c, v, v1


def main():
    pts, a, c, v, v1 = setup()
    print(f"pyjson 200点: ∠(v, v1) = {angle(v, v1):.1f}°  (PCA vs 理论焦点方向)")
    print(f"c={c:.3f}  a={a:.3f}")

    # 沿从 v1 到 v 的大圆扫描 (旋转轴 = v1×v)
    axis = np.cross(v1, v)
    if np.linalg.norm(axis) < 1e-9:
        axis = np.array([1.0, 0, 0])
    total = angle(v1, v)
    print("\n扫描 v5: 从 v1 (0°) 旋转到 v (全角), 记录 P(v5) 与两夹角")
    print(f"{'deg':>5s} {'∠(v5,v1)':>9s} {'∠(v5,v)':>8s} {'|Δ|':>6s} "
          f"{'P(v5)':>8s} {'P×100':>7s} {'|P-P(v1)|×100':>13s} {'|P-P(v)|×100':>12s}")
    steps = 41
    p_v1 = None
    for k in range(steps + 1):
        rot = total * k / steps
        v5 = rotate_vec(v1, axis, rot)
        v5 = v5 / np.linalg.norm(v5)
        s = batch_probability(pts, v5 * c, -v5 * c, a)[0]
        ang1 = angle(v5, v1)
        angv = angle(v5, v)
        if p_v1 is None:
            p_v1 = s
        p_v = batch_probability(pts, v * c, -v * c, a)[0]
        print(f"{rot:5.1f} {ang1:9.2f} {angv:8.2f} {abs(ang1-angv):6.2f} "
              f"{s:8.5f} {s*100:7.2f} {abs(s-p_v1)*100:13.2f} {abs(s-p_v)*100:12.2f}")
    print()
    print("注: 若 |P-P(v1)|×100 ≈ ∠(v5,v1) 或与 |Δ| 同步变化 → 概率可替代角度")
    print("DONE")


if __name__ == "__main__":
    main()
