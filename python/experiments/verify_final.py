# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
最终判定 (统一数据集, 修正数据构造不一致 bug):
  档1: PCA环 θ<0.5°                  → 成立
  档1失败: 0.5°≤θ≤30°                → 容错带 (不进档2)
  档1失败: θ>30°                     → 档2: 概率环 2000 次
      规则A(锁定): 迭代中 θ 曾 <30°  → t1 已锁定 → 成立(剔除,不判失败)
      规则B(固定范围): 全程 θ≥30° 时, R(2000)≤20° 且 R(2000)≤1.3*R(1000) → 成立
      否则 → 失败
三列: 原始角度(理论初值处θ) | PCA角度 | 概率环收敛角度
"""
import numpy as np

import nsjy_algorithms as m
import verify_probability_correction as vp
import verify_fixedpoint as vf

t10, t20 = m.compute_taus()

# 档2 准入案例 (θ_PCA>30°, 与分类所用数据集一致)
TIER2 = {"ball0": 52.568, "ball4": 33.340, "ball6": 42.0, "ball7": 42.8,
         "ellip2": 32.955, "ellip3": 37.342, "ellipN2": 39.683, "gauss2": 50.794}


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
    """与 θ_PCA 分类同一数据源: 批次案例用 vp.build, 新案例用 make_set"""
    if name in vp.BATCH:
        return vp.build(name)
    tag = name.rstrip("0123456789")
    seed = int(name[len(tag):])
    return make_set(tag, seed)


print("=" * 122)
print("档2 判定 (概率环 2000 次, 每10次采样): 规则A=迭代中θ<30°→锁定成立; "
      "规则B=固定范围R2000≤20°")
print("=" * 122)
hdr = (f"{'name':8s} {'原始角θ0':>8s} {'PCA角θpca':>9s} | {'minθ':>6s} {'θend':>6s} "
       f"{'R500':>6s} {'R1000':>7s} {'R2000':>7s} | {'判定':>16s}")
print(hdr)
summary = {}
for name, th_pca in sorted(TIER2.items()):
    pts = get_pts(name)
    r30, r45, a, axis = vp.prepare(pts)
    hist = vf.prob_loop(pts, r30, r45, a, t10, t20, n_iter=2000, every=10)
    angs = np.array([vp.ang_line(dn, axis) if dn is not None else float("nan")
                     for _, _, _, dn in hist])
    th_orig = angs[0]
    th_min = float(np.nanmin(angs))
    th_end = float(angs[-1])
    it_min = int(hist[int(np.nanargmin(angs))][0])

    def rng_of(kiter):
        ks = max(1, kiter // 10)
        seg = angs[-ks:]
        return float(np.nanmax(seg) - np.nanmin(seg))

    R500 = rng_of(500)
    R1000 = rng_of(1000)
    R2000 = rng_of(2000)
    lo = float(np.nanmin(angs[-100:]))
    hi = float(np.nanmax(angs[-100:]))

    if th_min < 30.0:
        verdict = f"锁定(θmin={th_min:.1f}°<30)"
    elif R2000 <= 20.0 and R2000 <= 1.3 * max(R1000, 1e-9):
        verdict = f"固定范围[{lo:.0f},{hi:.0f}]"
    else:
        verdict = "失败"
    summary[name] = verdict
    print(f"{name:8s} {th_orig:8.1f} {th_pca:9.1f} | {th_min:6.1f} {th_end:6.1f} "
          f"{R500:6.1f} {R1000:7.1f} {R2000:7.1f} | {verdict:>16s}"
          + (f"  (minθ 出现在 it={it_min})" if th_min < 30 else ""))

print()
print("=" * 122)
print("汇总")
print("=" * 122)
locked = [k for k, v in summary.items() if v.startswith("锁定")]
fixed = [k for k, v in summary.items() if v.startswith("固定范围")]
failed = [k for k, v in summary.items() if v == "失败"]
print(f"规则A 锁定(迭代中<30°, 已锁定t1): {len(locked)} 个  {locked}")
print(f"规则B 固定范围:                   {len(fixed)} 个  {fixed}")
print(f"失败:                             {len(failed)} 个  {failed}")
print()
print("说明: 原始角θ0 = 理论初值 (t1,t2)=(-0.621,0.485),(-0.500,0.500) 处的角度;")
print("      PCA角θpca = PCA驱动环终点角度;  概率环 = 无v3的总概率加权修正环。")
print("DONE")
