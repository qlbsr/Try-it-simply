# -*- coding: utf-8 -*-
"""
全点集两套判断验证 (52 个数据集):
  套1 PCA环: θ<0.5°→档1成立 | 0.5~30°→容错带 | >30°→进档2
  套2 自迭代(概率环2000次): 规则A minθ<30°→锁定 | 规则B 固定范围→成立 | 否则失败
每个数据集必须给出明确判定, 无异常/无 NaN 遗漏
"""
import math
import time

import numpy as np

import nsjy_algorithms as m
import verify_probability_correction as vp
import verify_fixedpoint as vf

t10, t20 = m.compute_taus()


def make_set(tag, seed, n=200):
    if tag == "ball":
        return m.random_points(n, seed=seed)
    if tag == "gauss":
        rng = np.random.default_rng(1000 + seed)
        return [rng.standard_normal(3) * (0.5 + 2.0 * rng.random()) for _ in range(n)]
    if tag == "ellip":
        rng = np.random.default_rng(2000 + seed)
        u = rng.standard_normal(3)
        u = u / np.linalg.norm(u)
        return vp.make_ellipsoid_points(u, n, 1000 + seed, noise=0.02)
    if tag == "ellipN":
        rng = np.random.default_rng(3000 + seed)
        u = rng.standard_normal(3)
        u = u / np.linalg.norm(u)
        return vp.make_ellipsoid_points(u, n, 1000 + seed, noise=0.1)
    raise ValueError(tag)


def get_pts(name):
    if name in vp.BATCH:
        return vp.build(name)
    tag = name.rstrip("0123456789")
    seed = int(name[len(tag):])
    return make_set(tag, seed)


def tier2_check(pts, r30, r45, a, axis):
    """套2: 概率环 2000 次, 返回 (判定, minθ)"""
    hist = vf.prob_loop(pts, r30, r45, a, t10, t20, n_iter=2000, every=10)
    angs = np.array([vp.ang_line(dn, axis) if dn is not None else float("nan")
                     for _, _, _, dn in hist])
    th_min = float(np.nanmin(angs))
    if th_min < 30.0:
        return f"档2:锁定({th_min:.1f}°<30)", th_min
    seg100 = angs[-100:]
    seg200 = angs[-200:]
    R1000 = float(np.nanmax(seg100) - np.nanmin(seg100))
    R2000 = float(np.nanmax(seg200) - np.nanmin(seg200))
    if R2000 <= 20.0 and R2000 <= 1.3 * max(R1000, 1e-9):
        return (f"档2:固定范围[{np.nanmin(seg200):.0f},{np.nanmax(seg200):.0f}]", th_min)
    return "档2:失败", th_min


names = ([f"ball{i}" for i in range(14)] + [f"gauss{i}" for i in range(14)]
         + [f"ellip{i}" for i in range(12)] + [f"ellipN{i}" for i in range(12)])

print("=" * 100)
print(f"两套判断 全点集验证: {len(names)} 个数据集")
print("=" * 100)
print(f"{'name':8s} {'θ_PCA':>7s} {'判定':>24s}  用时")
rows = []
t_start = time.time()
for i, name in enumerate(names):
    t0 = time.time()
    try:
        pts = get_pts(name)
        r30, r45, a, axis = vp.prepare(pts)
        t1f, t2f, F1f, F2f = m.refine_moduli_by_axis(pts, t10, t20, axis,
                                                     max_iter=40, verbose=False)
        d = F1f - F2f
        nrm = np.linalg.norm(d)
        th = vp.ang_line(d / nrm, axis) if nrm > 1e-9 else float("nan")
        if math.isnan(th):
            verdict = "异常:焦点退化"
        elif th < 0.5:
            verdict = "档1成立"
        elif th <= 30.0:
            verdict = "容错带"
        else:
            verdict, _ = tier2_check(pts, r30, r45, a, axis)
        rows.append((name, th, verdict))
    except Exception as e:
        rows.append((name, float("nan"), f"异常:{type(e).__name__}:{e}"))
    dt = time.time() - t0
    if i % 8 == 7:
        print(f"  ... 进度 {i+1}/{len(names)}  累计 {(time.time()-t_start)/60:.1f} min")

print(f"{'name':8s} {'θ_PCA':>7s} {'判定':>24s}")
for name, th, verdict in rows:
    print(f"{name:8s} {th:7.2f} {verdict:>24s}")

print()
print("=" * 100)
print("汇总")
print("=" * 100)
t1 = sum(1 for _, _, v in rows if v == "档1成立")
tb = sum(1 for _, _, v in rows if v == "容错带")
t2 = sum(1 for _, _, v in rows if v.startswith("档2:"))
exc = sum(1 for _, _, v in rows if v.startswith("异常"))
lock = sum(1 for _, _, v in rows if "锁定" in v)
fixr = sum(1 for _, _, v in rows if "固定范围" in v)
fail = sum(1 for _, _, v in rows if v == "档2:失败")
print(f"档1成立: {t1}   容错带: {tb}   进档2: {t2}   (异常/未判定: {exc})")
print(f"档2明细: 锁定 {lock} | 固定范围 {fixr} | 失败 {fail}")
print(f"总运行时间 {(time.time()-t_start)/60:.1f} min")
print("DONE")
