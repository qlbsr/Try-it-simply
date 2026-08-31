# -*- coding: utf-8 -*-
"""
档2 判据 v2: "自收敛到固定范围"
  跑概率环 1000 次, 对 θ 做累计窗口范围分析:
    R(k) = 最后 k 次迭代的 θ 范围 (max-min)
  判定: R(1000) ≤ 20° 且 R(1000) ≤ 1.5*R(500) → 落在固定范围 [min,max] → 成立
  中途逃逸(全范围 >30°) 或 R(1000)>20° → 失败
"""
import numpy as np

import nsjy_algorithms as m
import verify_probability_correction as vp
import verify_fixedpoint as vf

t10, t20 = m.compute_taus()

TIER2 = {  # 档2 准入案例 (θ_PCA>30°)
    "ball0": 52.568, "ball4": 33.340, "ball6": 42.0, "ball7": 42.8,
    "ellip2": 32.955, "ellip3": 37.342, "ellipN2": 39.683, "gauss2": 50.794,
}


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


print("=" * 116)
print("概率环 1000 次: θ 每 10 次采样, 累计窗口范围 R(100/200/500/1000 次迭代)")
print("=" * 116)
hdr = (f"{'name':8s} {'θ_PCA':>7s} | {'θ0':>5s} {'θ1000':>6s} | "
       f"{'R100':>5s} {'R200':>5s} {'R500':>5s} {'R1000':>6s} | "
       f"{'全程带':>10s} {'判定':>5s}")
print(hdr)
summary = {}
for name, th_pca in sorted(TIER2.items()):
    tag = name.rstrip("0123456789")
    seed = int(name[len(tag):])
    pts = make_set(tag, seed)
    r30, r45, a, axis = vp.prepare(pts)
    hist = vf.prob_loop(pts, r30, r45, a, t10, t20, n_iter=1000, every=10)
    angs = np.array([vp.ang_line(dn, axis) if dn is not None else float("nan")
                     for _, _, _, dn in hist])
    n = len(angs)
    R = {}
    for kiter in (100, 200, 500, 1000):
        ksamp = max(1, kiter // 10)
        seg = angs[-ksamp:]
        R[kiter] = (float(seg.max() - seg.min()), float(seg.min()), float(seg.max()))
    r1000, lo1000, hi1000 = R[1000]
    r500, _, _ = R[500]
    full_lo, full_hi = float(angs.min()), float(angs.max())
    escape = (full_hi - full_lo) > 30.0
    bounded = r1000 <= 20.0 and r1000 <= 1.5 * max(r500, 1e-9)
    verdict = "成立" if (bounded and not escape) else "失败"
    summary[name] = verdict
    print(f"{name:8s} {th_pca:7.1f} | {angs[0]:5.1f} {angs[-1]:6.1f} | "
          f"{R[100][0]:5.1f} {R[200][0]:5.1f} {R[500][0]:5.1f} {r1000:6.1f} | "
          f"[{full_lo:.1f},{full_hi:.1f}]  {verdict}")

print()
print(f"档2 判定汇总: 成立 {sum(1 for v in summary.values() if v=='成立')}/"
      f"{len(summary)}   各案例: {summary}")
print()
print("=== 补充: 档1已收敛案例的 概率环 落点 (同一数据集, 档1与档2是否一致) ===")
for name in ("ellip0", "gauss0"):
    pts = vp.build(name)
    r30, r45, a, axis = vp.prepare(pts)
    hist = vf.prob_loop(pts, r30, r45, a, t10, t20, n_iter=500, every=25)
    angs = [vp.ang_line(dn, axis) if dn is not None else float("nan")
            for _, _, _, dn in hist]
    print(f"{name}: PCA环 θ={vp.BATCH[name][2]:.2f}°  概率环 θ500={angs[-1]:.1f}°  "
          f"尾100={[round(a,1) for a in angs[-4:]]}")
print("DONE")
