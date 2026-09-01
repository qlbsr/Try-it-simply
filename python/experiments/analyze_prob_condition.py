# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
最终: 沿大圆扫描 d (模拟迭代轨迹), 找 |Δ| 与概率差组合的最佳替代条件式
  数据: pyjson 真实 200 点 + 多个合成集
  组合候选 (全部不依赖 PCA):
    c15=|s1-s5|×100, c25=|s2-s5|×100, c35=|s3-s5|×100, c45=|s4-s5|×100
    c12=|s1-s2|, c13=|s1-s3|, c14=|s1-s4|, c23, c24, c34
    线性组合: a·c15 + b·c35 等
  判据: corr(组合, |Δ|) 最高者 → 用 组合 ≤ 阈值 替代 |Δ| ≤ 16
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


def dataset(name, points):
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
    f1, f2 = m.extract_foci(pts, pt, sig[1], pt, sig[2])
    pt01, p101, p201, sig01 = dd.fast_probs(r30, r45, t_, t_, a)
    F01, F02 = m.extract_foci(pts, p101, sig01[1], p201, sig01[2])
    f01, f02 = m.extract_foci(pts, pt01, sig01[1], pt01, sig01[2])

    def norm(vv):
        return np.asarray(vv, float) / np.linalg.norm(vv)

    v = m.pca(pts)[1]
    v = v / np.linalg.norm(v)
    v1, v2, v3, v4 = norm(F1 - F2), norm(f1 - f2), norm(F01 - F02), norm(f01 - f02)
    refs = {"s1": v1, "s2": v2, "s3": v3, "s4": v4}
    svals = {k: batch_probability(P, vv * c, -vv * c, a)[0] for k, vv in refs.items()}

    # 沿 v1→v 大圆扫描 41 步
    axis = np.cross(v1, v)
    if np.linalg.norm(axis) < 1e-9:
        axis = np.array([1.0, 0, 0])
    total = angle(v1, v)
    rows = []
    for k in range(41):
        rot = total * k / 40
        d = rotate_vec(v1, axis, rot)
        d = d / np.linalg.norm(d)
        ad = angle(d, v)
        ad1 = angle(d, v1)
        delta = abs(ad - ad1)
        s5 = batch_probability(P, d * c, -d * c, a)[0]
        row = {"delta": delta}
        for kk, sval in svals.items():
            row[f"c{kk[1]}5"] = abs(sval - s5) * 100
        # 参考间差 (常数)
        row["c12"] = abs(svals["s1"] - svals["s2"]) * 100
        row["c13"] = abs(svals["s1"] - svals["s3"]) * 100
        row["c14"] = abs(svals["s1"] - svals["s4"]) * 100
        row["c23"] = abs(svals["s2"] - svals["s3"]) * 100
        row["c24"] = abs(svals["s2"] - svals["s4"]) * 100
        row["c34"] = abs(svals["s3"] - svals["s4"]) * 100
        rows.append(row)
    return name, rows


def main():
    datasets = {}
    with open(PYJSON, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    datasets["pyjson"] = np.array([[p["x"], p["y"], p["z"]] for p in raw], float)
    for seed in range(3):
        datasets[f"ellip{seed}"] = np.array(dd.make_ellip(seed), float)
    for seed in range(2):
        datasets[f"ball{seed}"] = np.array(dd.make_ball(seed), float)

    all_rows = []
    names = []
    for name, pts in datasets.items():
        nm, rows = dataset(name, pts)
        all_rows.append(rows)
        names.append(nm)

    # 合并所有数据集的扫描点
    keys = ["c15", "c25", "c35", "c45", "c12", "c13", "c14", "c23", "c24", "c34"]
    D = {k: [] for k in keys}
    DEL = []
    for rows in all_rows:
        for r in rows:
            DEL.append(r["delta"])
            for k in keys:
                D[k].append(r[k])
    DEL = np.array(DEL)
    D = {k: np.array(v) for k, v in D.items()}

    print("=" * 84)
    print(f"扫描点 {len(DEL)} 个 (6 数据集 × 41 步): corr(组合, |Δ|)")
    print("=" * 84)
    print(f"{'组合':8s} {'corr':>7s} {'|Δ|≤16时 组合值域':>22s} {'|Δ|>16时 组合值域':>22s}")
    for k in keys:
        corr = np.corrcoef(DEL, D[k])[0, 1] if D[k].std() > 0 else float('nan')
        lo16, hi16 = D[k][DEL <= 16].min(), D[k][DEL <= 16].max()
        logt, higt = D[k][DEL > 16].min(), D[k][DEL > 16].max()
        print(f"{k:8s} {corr:+7.3f} {lo16:10.2f}~{hi16:9.2f} {logt:12.2f}~{higt:9.2f}")

    # 线性组合 a·c15 + b·c35 的网格搜索
    print()
    print("=" * 84)
    print("线性组合搜索: min |a·c15 + b·c35 - |Δ||")
    print("=" * 84)
    best = None
    for a in np.arange(0, 3.01, 0.1):
        for b in np.arange(0, 3.01, 0.1):
            if a == 0 and b == 0:
                continue
            combo = a * D["c15"] + b * D["c35"]
            # 用前 4 数据集拟合, 后 2 验证 (简化: 全部拟合看残差)
            err = np.mean(np.abs(combo - DEL))
            if best is None or err < best[0]:
                best = (err, a, b)
    err, a, b = best
    combo = a * D["c15"] + b * D["c35"]
    print(f"最佳: |Δ| ≈ {a:.1f}·|s1-s5| + {b:.1f}·|s3-s5| (×100)  残差均值 {err:.2f}°")
    corr = np.corrcoef(DEL, combo)[0, 1]
    print(f"corr(combo, |Δ|) = {corr:+.3f}")
    # 替代阈值: 找 combo 阈值使 |Δ|≤16 与 combo≤T 一致率最高
    best_acc, best_t = 0, 0
    for T in np.arange(0.5, 40, 0.5):
        pred = combo <= T
        true = DEL <= 16
        acc = (pred == true).mean()
        if acc > best_acc:
            best_acc, best_t = acc, T
    print(f"替代条件: combo ≤ {best_t:.1f} ⟺ |Δ|≤16   一致率 {best_acc*100:.1f}%")
    print()
    print("注: c15 与 c35 都用不依赖 PCA 的方向 (v1 理论焦点, v3 (0,1)焦点)")
    print("DONE")


if __name__ == "__main__":
    main()
