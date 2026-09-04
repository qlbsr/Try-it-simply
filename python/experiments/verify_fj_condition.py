# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
分析两个终止条件是否等价:
  C_new: Math.Abs(Math.Abs(fj) + Math.Abs(fj5) - 180) <= 10
         fj  = atan2(v.z, v.x)*Rad2Deg   [PCA 主轴 v 的 xz 平面方位角, 固定]
         fj5 = atan2(v5.z, v5.x)*Rad2Deg [当前方向 v5 的 xz 平面方位角]
  C_old: (angleDeg + angleDeg1)*0.5 - min(angleDeg, angleDeg1) <= 8
         angleDeg  = ∠(d, v)    [3D 夹角]
         angleDeg1 = ∠(d, v1)   [3D 夹角, v1=(F1-F2)]
数学: (a+b)/2-min = |a-b|/2 → C_old ⟺ |∠(d,v) - ∠(d,v1)| <= 16
      fj 是 2D 投影方位角(丢失 y 信息), 且不含 v1!
      → 几何上不太可能等价, 用数值验证
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


def azimuth_xz(vv):
    """atan2(z,x) 度, 与 C# 一致"""
    vv = np.asarray(vv, float) / np.linalg.norm(vv)
    return math.degrees(math.atan2(vv[2], vv[0]))


def main():
    with open(PYJSON, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    pts = [np.array([p["x"], p["y"], p["z"]], float) for p in raw]
    P = np.array(pts, float)
    rp = n2.compute_rp(pts)
    d2 = math.asin(math.cos(math.radians(30)) / math.pi)
    a0 = rp * (1 + math.sin(d2))
    _, r45, r30, _ = n2.yzqx(pts, rp)
    t1t, t2t = n2.compute_taus()
    t_ = complex(0, 1)
    pt, p1, p2, sig = dd.fast_probs(r30, r45, t1t, t2t, a0)
    F1, F2 = m.extract_foci(pts, p1, sig[1], p2, sig[2])
    f1, f2 = m.extract_foci(pts, pt, sig[1], pt, sig[2])
    pt01, p101, p201, sig01 = dd.fast_probs(r30, r45, t_, t_, a0)
    F01, F02 = m.extract_foci(pts, p101, sig01[1], p201, sig01[2])
    v = m.pca(pts)[1]
    v = v / np.linalg.norm(v)      # 与 C# pca() 一致 (v3)
    v1 = (np.array(F1, float) - np.array(F2, float)); v1 = v1 / np.linalg.norm(v1)
    fj = azimuth_xz(v)             # 固定

    # 模拟迭代: d 沿 v1→v 大圆扫描 (真实迭代 d 会这样转吗? 至少验证两条件的数值关系)
    # 更真实的: 用多方向采样覆盖球面, 看两条件的触发一致性
    print("=" * 96)
    print("fj/fj5 条件 vs 角度条件: 数值等价性 (pyjson)")
    print("=" * 96)
    print(f"v(pca)={np.round(v,3)} fj={fj:.1f}°  ∠(v,v1)={angle(v,v1):.1f}°")
    print(f"v1={np.round(v1,3)}  ∠(v1 与 xz 投影关系?)")
    print()

    # 沿 v1→v 大圆扫描 d (与之前测试一致的轨迹)
    axis = np.cross(v1, v)
    if np.linalg.norm(axis) < 1e-9:
        axis = np.array([1.0, 0, 0])
    total = angle(v1, v)
    print(f"{'rot':>5s} {'fj5':>8s} {'|fj|+|fj5|-180':>15s} {'new≤10?':>7s} | "
          f"{'|Δ|':>6s} {'cond_old':>8s} {'old≤8?':>6s}")
    agree = 0
    n_ = 0
    for k in range(41):
        rot = total * k / 40
        ang = math.radians(rot)
        d = (v1 * math.cos(ang) + np.cross(axis, v1) * math.sin(ang)
             + axis * np.dot(axis, v1) * (1 - math.cos(ang)))
        d = d / np.linalg.norm(d)
        ad = angle(d, v)
        ad1 = angle(d, v1)
        delta = abs(ad - ad1)
        cond_old = (ad + ad1) * 0.5 - min(ad, ad1)
        fj5 = azimuth_xz(d)
        cond_new = abs(abs(fj) + abs(fj5) - 180)
        new_ok = cond_new <= 10
        old_ok = cond_old <= 8
        agree += (new_ok == old_ok)
        n_ += 1
        if k % 4 == 0 or new_ok or old_ok:
            print(f"{rot:5.1f} {fj5:8.1f} {cond_new:15.2f} {str(new_ok):>7s} | "
                  f"{delta:6.1f} {cond_old:8.2f} {str(old_ok):>6s}")
    print()
    print(f"沿大圆 41 点: 两条件一致率 {agree}/{n_} = {agree/n_*100:.0f}%")
    print()

    # 球面随机采样 (更全面)
    rng = np.random.default_rng(7)
    dirs = rng.standard_normal((2000, 3))
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    agree2 = 0
    for d in dirs:
        ad = angle(d, v)
        ad1 = angle(d, v1)
        cond_old = (ad + ad1) * 0.5 - min(ad, ad1)
        cond_new = abs(abs(fj) + abs(azimuth_xz(d)) - 180)
        agree2 += ((cond_new <= 10) == (cond_old <= 8))
    print(f"球面随机 2000 方向: 两条件一致率 {agree2/2000*100:.1f}%")
    print("DONE")


if __name__ == "__main__":
    main()
