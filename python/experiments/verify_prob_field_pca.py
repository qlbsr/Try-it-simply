# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
用户洞察深入: 三角形平面族携带 PCA 信息 → 用概率场的统计量定位主轴
  核心: 对候选方向 v, 计算全部点的概率 s_i(v) = exp(-|d1+d2-2a|/(2a))
        概率场统计量 (如熵/方差/峰值) 应随 v 逼近主轴而变化
  无 PCA: 扫描方向空间, 找统计量的极值方向 = 数据主轴
  测试: ① 不同统计量 ② 扫描定位 ③ 对比 PCA 主轴
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


def prob_field(v, P, c, a):
    """全部点的概率数组 s_i(v)"""
    return batch_probability(P, v * c, -v * c, a)


def field_stats(s):
    """概率场的统计量"""
    return {
        "mean": s.mean(),
        "std": s.std(),
        "entropy": -np.sum(s * np.log(s + 1e-12)),
        "max": s.max(),
        "sharp": np.mean(s ** 2),      # 尖锐度
    }


def main():
    with open(PYJSON, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    pts = [np.array([p["x"], p["y"], p["z"]], float) for p in raw]
    P = np.array(pts, float)
    rp = n2.compute_rp(pts)
    d2 = math.asin(math.cos(math.radians(30)) / math.pi)
    a = rp * (1 + math.sin(d2))
    e3 = math.cos(d2) * 2 / (1 + math.sin(d2))
    c = rp * math.cos(d2) * e3
    v_pca = m.pca(pts)[1]
    v_pca = v_pca / np.linalg.norm(v_pca)

    # 扫描方向: 球面均匀网格 (斐波那契球)
    print("=" * 84)
    print("概率场统计量扫描: 找数据主轴 (无 PCA)")
    print("=" * 84)
    n_samp = 400
    phi = math.pi * (3 - math.sqrt(5))
    dirs = []
    for i in range(n_samp):
        y = 1 - (i / (n_samp - 1)) * 2
        r = math.sqrt(1 - y * y)
        th = phi * i
        dirs.append(np.array([math.cos(th) * r, y, math.sin(th) * r]))
    dirs = np.array(dirs)

    # 计算各统计量
    stats_keys = ["mean", "std", "entropy", "max", "sharp"]
    best = {k: (None, -1e18 if k in ("std", "entropy", "sharp", "max") else 1e18)
            for k in stats_keys}
    extremum = {"mean": "min", "std": "max", "entropy": "max", "max": "max", "sharp": "max"}
    for i, v in enumerate(dirs):
        s = prob_field(v, P, c, a)
        st = field_stats(s)
        for k in stats_keys:
            val = st[k]
            if extremum[k] == "max" and (best[k][1] is None or val > best[k][1]):
                best[k] = (v.copy(), val)
            if extremum[k] == "min" and (best[k][1] is None or val < best[k][1]):
                best[k] = (v.copy(), val)

    print(f"PCA 主轴: {np.round(v_pca,3)}")
    for k in stats_keys:
        v_b, val = best[k]
        print(f"  {k:8s} 极值方向 {np.round(v_b,3)}  ∠(极值,PCA)={angle(v_b, v_pca):6.1f}°  "
              f"值={val:.4f}")

    # 参考方向对照
    print()
    print("参考方向与 PCA 的夹角:")
    print(f"  v1(理论焦点) = {np.round(m.pca(pts)[0],3)}... 见下方")
    print()
    print("若某统计量极值方向 ≈ PCA 主轴 → 三角形平面族可定位主轴 (无 PCA)")
    print("DONE")


if __name__ == "__main__":
    main()
