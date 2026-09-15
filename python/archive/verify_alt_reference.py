# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
关键测试: 用"无 PCA 的对照场"替代 s0(PCA场) 的角色
  原条件: |angleDeg - angleDeg1| = |∠(d,v) - ∠(d,v1)| ≤ 16
  场代理: angleDeg ≈ f(corr(s5, s0))  [s0 需 PCA]
  替代:   s0 换成 s3 (v3=(0,1)焦点方向, 无 PCA) 或其他对照
  测试: |∠(d,v3) - ∠(d,v1)| 与 |∠(d,v) - ∠(d,v1)| 的等价性
        (若 v3 与 v 接近 → 可替代; 已知 ∠(v,v3)=80.7° 不接近, 验证)
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


def main():
    with open(PYJSON, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    pts = [np.array([p["x"], p["y"], p["z"]], float) for p in raw]
    rp = n2.compute_rp(pts)
    d2 = math.asin(math.cos(math.radians(30)) / math.pi)
    a = rp * (1 + math.sin(d2))
    _, r45, r30, _ = n2.yzqx(pts, rp)
    t1t, t2t = n2.compute_taus()
    t_ = complex(0, 1)
    pt, p1, p2, sig = dd.fast_probs(r30, r45, t1t, t2t, a)
    F1, F2 = m.extract_foci(pts, p1, sig[1], p2, sig[2])
    f1, f2 = m.extract_foci(pts, pt, sig[1], pt, sig[2])
    pt01, p101, p201, sig01 = dd.fast_probs(r30, r45, t_, t_, a)
    F01, F02 = m.extract_foci(pts, p101, sig01[1], p201, sig01[2])
    f01, f02 = m.extract_foci(pts, pt01, sig01[1], pt01, sig01[2])
    v = m.pca(pts)[1]
    v = v / np.linalg.norm(v)
    v1 = (np.array(F1, float) - np.array(F2, float)); v1 = v1 / np.linalg.norm(v1)
    v2 = (np.array(f1, float) - np.array(f2, float)); v2 = v2 / np.linalg.norm(v2)
    v3 = (np.array(F01, float) - np.array(F02, float)); v3 = v3 / np.linalg.norm(v3)
    v4 = (np.array(f01, float) - np.array(f02, float)); v4 = v4 / np.linalg.norm(v4)

    print("=" * 88)
    print("替代参考分析: 哪个对照方向最接近 PCA 主轴 v?")
    print("=" * 88)
    refs = {"v1(理论焦点)": v1, "v2(拟合焦点)": v2, "v3((0,1)焦点)": v3, "v4((0,1)拟合)": v4}
    for nm, vv in refs.items():
        print(f"  ∠({nm}, v_pca) = {angle(vv, v):6.1f}°")
    print()
    print("若某对照与 v 夹角 < 20° → 可用它替代 s0 (完全无 PCA)")
    print()

    # 沿 v1→v 大圆扫描: 比较 原条件 vs 各替代条件
    axis = np.cross(v1, v)
    if np.linalg.norm(axis) < 1e-9:
        axis = np.array([1.0, 0, 0])
    total = angle(v1, v)
    print("=" * 88)
    print("扫描: 原条件 |∠(d,v)-∠(d,v1)| vs 替代 |∠(d,vX)-∠(d,v1)|")
    print("=" * 88)
    print(f"{'rot':>5s} {'|Δ原|':>6s} | " + " ".join(f"{nm:>10s}" for nm in refs))
    for k in range(0, 41, 4):
        rot = total * k / 40
        ang = math.radians(rot)
        d = (v1 * math.cos(ang) + np.cross(axis, v1) * math.sin(ang)
             + axis * np.dot(axis, v1) * (1 - math.cos(ang)))
        d = d / np.linalg.norm(d)
        orig = abs(angle(d, v) - angle(d, v1))
        alts = [abs(angle(d, vv) - angle(d, v1)) for vv in refs.values()]
        print(f"{rot:5.1f} {orig:6.1f} | " + " ".join(f"{x:10.1f}" for x in alts))
    print()
    print("对照结论: 替代条件与原件在不同位置同号/反号 → 需一致性评估")
    print("DONE")


if __name__ == "__main__":
    main()
