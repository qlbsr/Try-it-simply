# -*- coding: utf-8 -*-
"""
验证用户假设: 对任意点集, 外环迭代中 angle_pca(与PCA夹角) 的趋势
始终向 angle1(与初始焦点方向夹角) 靠近 → 收敛条件 = 两角相等.

完全复刻 n2sjy2.py 主循环语义 (内环从 (i,i) 出发, 以当前焦点方向为内环轴,
PCA 只测量不参与更新), 仅把 compute_probabilities_from_taus 换成向量化等价实现.
"""
import math
import time

import numpy as np

import n2sjy2 as n2


def fast_probs(r30, r45, t1, t2, a, rng=20):
    """向量化版 compute_probabilities_from_taus (数学等价)"""
    Z1 = np.asarray(r30, complex)
    Z2 = np.asarray(r45, complex)
    mg = np.arange(-rng, rng + 1)
    MM, NN = np.meshgrid(mg, mg)
    L1 = MM.ravel() + NN.ravel() * t1
    L2 = MM.ravel() + NN.ravel() * t2
    d1 = np.abs(Z1[:, None] - L1[None, :]).min(axis=1)
    d2 = np.abs(Z2[:, None] - L2[None, :]).min(axis=1)
    s1 = np.sqrt(np.mean(d1 ** 2))
    s2 = np.sqrt(np.mean(d2 ** 2))
    p1 = np.exp(-d1 ** 2 / (2 * s1 ** 2))
    p2 = np.exp(-d2 ** 2 / (2 * s2 ** 2))
    pt = np.exp(-np.abs(d1 + d2 - 2 * a) / (2 * a))
    return pt, p1, p2, np.array([2 * a, s1, s2])


n2.compute_probabilities_from_taus = fast_probs


def run_outer(points, outer=120, inner=30):
    rp = n2.compute_rp(points)
    d2 = math.asin(math.cos(math.radians(30)) / math.pi)
    a = rp * (1 + math.sin(d2))
    t1_theory, t2_theory = n2.compute_taus()
    normalized_list, r45, r30, r74 = n2.yzqx(points, rp)

    prob_total, prob1, prob2, sigmas = n2.compute_probabilities_from_taus(
        r30, r45, t1_theory, t2_theory, a)
    F1_init, F2_init = n2.extract_foci(points, prob1, sigmas[1], prob2, sigmas[2])
    F1_fit, F2_fit = n2.fit_foci_by_probability(points, prob_total, F1_init, F2_init, a)
    f1, f2 = np.array(F1_fit, float), np.array(F2_fit, float)
    init_dir = (f1 - f2) / np.linalg.norm(f1 - f2)
    v_pca = n2.pca_axis(points)

    t1 = t2 = complex(0, 1)
    rec = []
    prev_dir = init_dir
    for i in range(outer):
        prob_total_new, _, _, _ = n2.compute_probabilities_from_taus(r30, r45, t1, t2, a)
        axis_now = (f1 - f2) / np.linalg.norm(f1 - f2)
        t1_new, t2_new, F1_new, F2_new = n2.refine_moduli_by_axis(
            points, r30, r45, a, complex(0, 1), complex(0, 1), axis_now, max_iter=inner)
        f1_new, f2_new = n2.fit_foci_by_probability(points, prob_total_new, F1_new, F2_new, a)
        dir_vec = (f1_new - f2_new) / np.linalg.norm(f1_new - f2_new)
        angle1 = math.degrees(math.acos(np.clip(np.dot(dir_vec, init_dir), -1, 1)))
        angle_pca = math.degrees(math.acos(np.clip(np.dot(dir_vec, v_pca), -1, 1)))
        d_stab = math.degrees(math.acos(np.clip(np.dot(dir_vec, prev_dir), -1, 1)))
        rec.append((i, angle1, angle_pca, d_stab))
        f1, f2 = f1_new, f2_new
        t1, t2 = t1_new, t2_new
        prev_dir = dir_vec
    return rec


def analyze(name, rec, outer=120):
    a1 = np.array([r[1] for r in rec])
    ap = np.array([r[2] for r in rec])
    dst = np.array([r[3] for r in rec])
    d = np.abs(ap - a1)                      # 两角之差
    trig = np.where(d <= 10.0)[0]            # 收敛条件 |Δ|≤10 触发轮次
    dec = np.mean(np.diff(d) < 0)            # 差减小的轮次占比
    d0, dend = d[0], d[-1]
    print(f"{name:8s} |Δ|0={d0:6.2f}°  |Δ|end={dend:6.2f}°  min={d.min():6.2f}°  "
          f"下降占比={dec:5.1%}  触发(|Δ|≤10)={len(trig):3d}/{len(d)}"
          + (f"  @it={trig[0]}" if len(trig) else "")
          + f"  方向稳定度末值={dst[-1]:5.2f}°")
    return d


def main():
    datasets = {}
    for s in range(3):
        rng = np.random.default_rng(100 + s)
        datasets[f"cube{s}"] = np.random.RandomState(10 + s).uniform(-1, 1, (200, 3))
    for s in range(3):
        rng = np.random.default_rng(200 + s)
        dirs = rng.standard_normal((200, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        datasets[f"ball{s}"] = [d * (rng.random() ** (1 / 3)) for d in dirs]
    for s in range(2):
        rng = np.random.default_rng(300 + s)
        u = rng.standard_normal(3)
        u /= np.linalg.norm(u)
        tmp = np.array([1.0, 0, 0]) if abs(u[0]) < 0.9 else np.array([0.0, 1, 0])
        v = np.cross(u, tmp)
        v /= np.linalg.norm(v)
        w = np.cross(u, v)
        th = rng.uniform(0, np.pi, 200)
        ph = rng.uniform(0, 2 * np.pi, 200)
        pts = []
        for i in range(200):
            pts.append(1.5 * np.sin(th[i]) * np.cos(ph[i]) * u
                       + 0.8 * np.sin(th[i]) * np.sin(ph[i]) * v
                       + 0.8 * np.cos(th[i]) * w + 0.02 * rng.standard_normal(3))
        datasets[f"ellip{s}"] = pts

    print("=" * 110)
    print("n2sjy2 外环: angle1(与初始方向) 与 angle_pca(与PCA) 的趋近验证 (outer=120, inner=30)")
    print("=" * 110)
    all_d = []
    for name, pts in datasets.items():
        t0 = time.time()
        rec = run_outer(pts, outer=120, inner=30)
        d = analyze(name, rec)
        all_d.append(d)
        print(f"    [{time.time()-t0:.0f}s]")
    print()
    print("跨数据集总结:")
    ends = [d[-1] for d in all_d]
    mins = [d.min() for d in all_d]
    fracs = [np.mean(np.diff(d) < 0) for d in all_d]
    trigs = [np.where(d <= 10.0)[0].size for d in all_d]
    print(f"  最终|Δ|: min={min(ends):.2f}°  mean={np.mean(ends):.2f}°  max={max(ends):.2f}°")
    print(f"  全程最小|Δ|: min={min(mins):.2f}°  mean={np.mean(mins):.2f}°  max={max(mins):.2f}°")
    print(f"  |Δ|下降轮次占比: min={min(fracs):.1%}  mean={np.mean(fracs):.1%}  max={max(fracs):.1%}")
    print(f"  触发收敛(|Δ|≤10°) 的点集数: {sum(1 for t in trigs if t>0)}/{len(datasets)}")
    print("DONE")


if __name__ == "__main__":
    main()
