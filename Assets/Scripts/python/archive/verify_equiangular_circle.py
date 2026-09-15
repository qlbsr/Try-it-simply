# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
用户数据 (117.9937, 110.0077, 65.46885) = (∠(d,v), ∠(d,F1-F2), ∠(vj,v)):
  即使 |Δ| 大 (110/117), 与角平分线夹角 65° 完全不同 → 迭代求的是什么?

几何答案: 迭代求 d 落在"等角大圆" {d : d·(v-w)=0} 上 (∠(d,v)=∠(d,w)),
          角平分线 vj ∝ v+w 只是该大圆上的一点 (解析满足 vj·(v-w)=0)。
  这里 w = F1-F2 (初始焦点方向), v = PCA 轴。
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


def load():
    with open(PYJSON, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    pts = [np.array([p["x"], p["y"], p["z"]], float) for p in raw]
    t1t, t2t = n2.compute_taus()
    rp = n2.compute_rp(pts)
    d2 = math.asin(math.cos(math.radians(30)) / math.pi)
    a = rp * (1 + math.sin(d2))
    _, r45, r30, _ = n2.yzqx(pts, rp)
    v = m.pca(pts)[1]
    v = v / np.linalg.norm(v)
    pt, p1, p2, sig = dd.fast_probs(r30, r45, t1t, t2t, a)
    F1, F2 = m.extract_foci(pts, p1, sig[1], p2, sig[2])
    return pts, r30, r45, a, v, F1, F2


def direction_at(t1, t2, pts, r30, r45, a):
    pt, p1, p2, sig = dd.fast_probs(r30, r45, t1, t2, a)
    F1, F2 = m.extract_foci(pts, p1, sig[1], p2, sig[2])
    d = np.array(F1, float) - np.array(F2, float)
    return d / np.linalg.norm(d)


def main():
    pts, r30, r45, a, v, F1, F2 = load()
    w = np.array(F1, float) - np.array(F2, float)
    w = w / np.linalg.norm(w)
    vj = v + w                      # 角平分线 (用户代码: v̂+d̂)
    vj = vj / np.linalg.norm(vj)

    print(f"v = PCA 轴;  w = (F1-F2)/|F1-F2| (初始焦点方向)")
    print(f"∠(v, w) = {angle(v, w):.2f}°  → 角平分线 vj 与 v 夹角 = ∠(v,w)/2 = {angle(v,w)/2:.2f}°")
    print(f"解析检查: vj·(v-w) = {np.dot(vj, v - w):.2e}  (角平分线必在等角大圆上, 恒为0)")
    print()

    # ---- 用户给出的迭代中间值 ----
    a_u, a1_u, avj_u = 117.9937, 110.0077, 65.46885
    print("=" * 70)
    print("用户数据 (117.99, 110.01, 65.47) = (∠(d,v), ∠(d,w), ∠(vj,v))")
    print("=" * 70)
    print(f"  |Δ| = |{a_u:.2f} − {a1_u:.2f}| = {abs(a_u-a1_u):.2f}° (迭代中途, 未收敛到等角大圆)")
    print(f"  ∠(vj,v) = {avj_u:.2f}° 应等于 ∠(v,w)/2 = {angle(v,w)/2:.2f}° "
          f"(一致度 {abs(avj_u - angle(v,w)/2):.2f}°)")
    # 若 d 在等角大圆上: cos∠(d,v) = cos∠(d,w) → d·(v-w) = 0
    lhs = math.cos(math.radians(a_u)) - math.cos(math.radians(a1_u))
    print(f"  等角大圆判据 cos∠(d,v)−cos∠(d,w) = {lhs:+.4f} "
          f"(→0 表示 d 在该圆上; 当前 |Δ|=8° 所以未到 0)")
    print()

    # ---- 收敛后的状态 (上次 NM 达成 |Δ|≈0 的 t1,t2) ----
    print("=" * 70)
    print("收敛状态验证 (|Δ|→0 时的 d 与等角大圆)")
    print("=" * 70)
    for label, t1, t2 in [("理论 taus", *n2.compute_taus()),
                          ("迭代0后", complex(-0.5, 0.9021), complex(-0.5, 1.5)),
                          ("达成时", complex(0.1704, 0.5798), complex(-0.5, 0.5))]:
        d = direction_at(t1, t2, pts, r30, r45, a)
        ad = angle(d, v)
        ad1 = angle(d, w)
        dvw = np.dot(d, v - w)
        print(f"  {label:8s}: ∠(d,v)={ad:7.2f}° ∠(d,w)={ad1:7.2f}° |Δ|={abs(ad-ad1):5.2f}° "
              f"d·(v−w)={dvw:+.4f}  ∠(d,vj)={angle(d,vj):6.1f}°")
    print()
    print("结论: |Δ|→0 ⟺ d·(v−w)→0 ⟺ d 落在等角大圆上;")
    print("      角平分线 vj 是该圆上一点 (恒满足 vj·(v−w)=0), 但不是迭代目标本身。")
    print("DONE")


if __name__ == "__main__":
    main()
