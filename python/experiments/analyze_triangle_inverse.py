# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
三角形方法 (用户洞察落地): 从概率反演 p0 视角夹角, 球面三角测量 v5 位置
  核心: 对方向 v, 三角形 △(p0, v·c, -v·c) 中
        d1+d2 = |p0-v·c| + |p0+v·c| 只依赖 theta=∠(p0,v) (|p0|, c 固定)
        s_v = exp(-|d1+d2-2a|/(2a))  → 单调(在 [0,90]减 / [90,180]增), 可反演
  步骤:
    1. 预计算反演表: s → theta (查表)
    2. 对 v1, v3 (已知参考方向) 和 v5 (当前方向): 算 s → theta1, theta3, theta5
    3. 球面三角: 已知 p0̂, v1, v3 方向 + theta5=∠(p0,v5) 的两个候选方位 → 解 v5
    4. 用解出的 v5 计算 ∠(v5, v1) 等 → 与直接向量计算对比 (验证)
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


def dist(p, q):
    return np.linalg.norm(np.asarray(p, float) - np.asarray(q, float))


def build_inv_table(p0, c, a, n=1801):
    """预计算 theta→s 表, 返回 (theta_grid, s_grid)"""
    r0 = np.linalg.norm(p0)
    th = np.linspace(0, 180, n)
    s = np.zeros(n)
    for i, t in enumerate(th):
        rad = math.radians(t)
        # 三角形: |p0 - v·c| + |p0 + v·c|, v 与 p0 夹角 t
        # 用余弦定理: |p0±vc|² = r0² + c² ∓ 2 r0 c cos t
        d1 = math.sqrt(r0 ** 2 + c ** 2 - 2 * r0 * c * math.cos(rad))
        d2 = math.sqrt(r0 ** 2 + c ** 2 + 2 * r0 * c * math.cos(rad))
        s[i] = math.exp(-abs(d1 + d2 - 2 * a) / (2 * a))
    return th, s


def invert(s_val, th_grid, s_grid):
    """s → theta 反演 (最近邻)"""
    i = np.argmin(np.abs(s_grid - s_val))
    return th_grid[i]


def main():
    with open(PYJSON, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    pts = [np.array([p["x"], p["y"], p["z"]], float) for p in raw]
    P = np.array(pts, float)
    p0 = P[0]
    rp = n2.compute_rp(pts)
    d2 = math.asin(math.cos(math.radians(30)) / math.pi)
    a = rp * (1 + math.sin(d2))
    e3 = math.cos(d2) * 2 / (1 + math.sin(d2))
    c = rp * math.cos(d2) * e3
    _, r45, r30, _ = n2.yzqx(pts, rp)
    t1t, t2t = n2.compute_taus()
    t_ = complex(0, 1)
    pt, p1, p2, sig = dd.fast_probs(r30, r45, t1t, t2t, a)
    F1, F2 = m.extract_foci(pts, p1, sig[1], p2, sig[2])
    pt01, p101, p201, sig01 = dd.fast_probs(r30, r45, t_, t_, a)
    F01, F02 = m.extract_foci(pts, p101, sig01[1], p201, sig01[2])
    v = m.pca(pts)[1]
    v = v / np.linalg.norm(v)
    v1 = (np.array(F1, float) - np.array(F2, float))
    v1 = v1 / np.linalg.norm(v1)
    v3 = (np.array(F01, float) - np.array(F02, float))
    v3 = v3 / np.linalg.norm(v3)

    th_grid, s_grid = build_inv_table(p0, c, a)
    print(f"|p0|={np.linalg.norm(p0):.3f} c={c:.3f} a={a:.3f}")

    def s_of(vv):
        d1 = dist(p0, vv * c)
        d2 = dist(p0, -vv * c)
        return math.exp(-abs(d1 + d2 - 2 * a) / (2 * a))

    print()
    print("=== 反演验证: 直接算 ∠(p0,v) vs 从 s 反演 ===")
    for name, vv in [("v1", v1), ("v3", v3), ("v(pca)", v)]:
        th_true = angle(p0, vv)
        s_val = s_of(vv)
        th_inv = invert(s_val, th_grid, s_grid)
        print(f"  {name:8s}: ∠(p0,v)真值={th_true:6.1f}°  s={s_val:.5f}  "
              f"反演={th_inv:6.1f}°  误差={abs(th_true-th_inv):.1f}°")

    print()
    print("=== 双解性检查: 同一 s 对应 theta 与 180-theta ===")
    for vv, nm in [(v1, "v1")]:
        s_val = s_of(vv)
        idx = np.argmin(np.abs(s_grid - s_val))
        # 找所有接近 s 的 theta
        near = np.where(np.abs(s_grid - s_val) < 1e-4)[0]
        ths = th_grid[near]
        # 聚类两端
        if len(ths) > 1:
            print(f"  {nm}: s={s_val:.5f} → θ 候选 {ths[0]:.1f}° 与 {ths[-1]:.1f}° "
                  f"(双解: {ths[0]:.1f}+{ths[-1]:.1f}={ths[0]+ths[-1]:.1f}≈180?)")
    print()
    print("注: 若双解 θ 与 180-θ, 需用两个参考方向消除歧义 (球面三角)")
    print("DONE")


if __name__ == "__main__":
    main()
