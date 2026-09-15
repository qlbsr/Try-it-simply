# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
系统搜索: 哪个概率组合能替代终止条件 cond = |angleDeg - angleDeg1|/2 ≤ 8
  angleDeg = ∠(v5, v)     [依赖 PCA]
  angleDeg1 = ∠(v5, v1)   [不依赖 PCA: v1=(F1-F2)]
  候选参考方向: v1(理论焦点) v2(拟合焦点) v3((0,1)焦点) v4((0,1)拟合) v5(当前)
  s_i = BatchProbability(v_i·c, -v_i·c)[0]
  对多数据集: 计算所有 (i,j) 对: |s_i - s_j|×100 与 ∠(v_i,v_j) 的相关系数
  以及: 能否用 s1..s5 (无 PCA) 重建 cond
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


def collect(name, points):
    """计算 v1..v5, s1..s5 (无 PCA 的 5 方向), 返回向量与概率"""
    pts = [np.array(p, float) for p in points]
    rp = n2.compute_rp(pts)
    d2 = math.asin(math.cos(math.radians(30)) / math.pi)
    a = rp * (1 + math.sin(d2))
    e3 = math.cos(d2) * 2 / (1 + math.sin(d2))
    h = rp * math.cos(d2)
    c = h * e3
    _, r45, r30, _ = n2.yzqx(pts, rp)
    t1t, t2t = n2.compute_taus()
    t_ = complex(0, 1)
    P = np.array(points, float)

    pt, p1, p2, sig = dd.fast_probs(r30, r45, t1t, t2t, a)
    F1, F2 = m.extract_foci(pts, p1, sig[1], p2, sig[2])
    f1, f2 = m.extract_foci(pts, pt, sig[1], pt, sig[2])   # 用 probTotal 拟合
    pt01, p101, p201, sig01 = dd.fast_probs(r30, r45, t_, t_, a)
    F01, F02 = m.extract_foci(pts, p101, sig01[1], p201, sig01[2])
    f01, f02 = m.extract_foci(pts, pt01, sig01[1], pt01, sig01[2])

    def norm(vv):
        return np.asarray(vv, float) / np.linalg.norm(vv)

    vs = {"v1(thF)": norm(F1 - F2), "v2(fitF)": norm(f1 - f2),
          "v3(iF)": norm(F01 - F02), "v4(ifitF)": norm(f01 - f02),
          "v5(cur=fitF)": norm(f1 - f2)}
    ss = {k: batch_probability(P, vv * c, -vv * c, a)[0] for k, vv in vs.items()}
    v = m.pca(pts)[1]
    v = v / np.linalg.norm(v)
    return name, vs, ss, v


def main():
    datasets = {}
    with open(PYJSON, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    datasets["pyjson"] = np.array([[p["x"], p["y"], p["z"]] for p in raw], float)
    for seed in range(3):
        datasets[f"ellip{seed}"] = np.array(dd.make_ellip(seed), float)
    for seed in range(2):
        datasets[f"ball{seed}"] = np.array(dd.make_ball(seed), float)

    rows = []
    for name, pts in datasets.items():
        rows.append(collect(name, pts))

    print("=" * 96)
    print("各方向概率 s_i (×100) 与方向夹角 (每数据集)")
    print("=" * 96)
    keys = list(rows[0][1].keys())
    for name, vs, ss, v in rows:
        print(f"  {name:8s} " + " ".join(f"{k}={ss[k]*100:6.2f}" for k in keys))
    print()

    print("=" * 96)
    print("成对验证: |s_i - s_j|×100 vs ∠(v_i, v_j)   [跨数据集相关性]")
    print("=" * 96)
    print(f"{'pair':22s} {'∠范围':>14s} {'|Δs|×100范围':>14s} {'corr':>7s} {'斜率(Δs/°)':>10s}")
    pairs = [(i, j) for i in range(len(keys)) for j in range(i + 1, len(keys))]
    for i, j in pairs:
        angs, ds = [], []
        for name, vs, ss, v in rows:
            angs.append(angle(vs[keys[i]], vs[keys[j]]))
            ds.append(abs(ss[keys[i]] - ss[keys[j]]) * 100)
        angs, ds = np.array(angs), np.array(ds)
        corr = np.corrcoef(angs, ds)[0, 1] if ds.std() > 0 else float('nan')
        slope = np.polyfit(angs, ds, 1)[0] if len(angs) > 2 else float('nan')
        print(f"  ({keys[i]},{keys[j]}) {angs.min():6.1f}~{angs.max():5.1f}° "
              f"{ds.min():8.2f}~{ds.max():7.2f} {corr:+7.3f} {slope:10.4f}")
    print()
    print("注: corr 高 + 斜率稳定 → 该概率差可替代对应夹角")
    print("DONE")


if __name__ == "__main__":
    main()
