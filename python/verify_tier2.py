# -*- coding: utf-8 -*-
"""
按修正后的门控重新判定:
  档1: PCA环 θ<0.5°                      → 成立
  档1失败 + 0.5°≤θ≤30°                    → 不进档2 (容错带; 附 t1,t2 对称性数据)
  档1失败 + θ>30°                         → 进档2: 概率环500次, 后100次角度范围≤3° 判定
收集所有 θ>30° 案例 (已有批次 + 新扫描) 跑档2
"""
import numpy as np

import nsjy_algorithms as m
import verify_probability_correction as vp
import verify_fixedpoint as vf

t10, t20 = m.compute_taus()

# θ_PCA: 已测批次 (verify_falsify 扫描结果)
THETA_PCA = {
    "ball0": 52.568, "ball1": 29.309, "ball2": 0.471, "ball3": 0.302, "ball4": 33.340,
    "gauss0": 0.091, "gauss1": 0.119, "gauss2": 50.794, "gauss3": 0.498, "gauss4": 2.305,
    "ellip0": 0.045, "ellip1": 21.958, "ellip2": 32.955, "ellip3": 37.342, "ellip4": 19.908,
    "ellipN0": 17.232, "ellipN1": 28.485, "ellipN2": 39.683, "ellipN3": 0.214, "ellipN4": 1.714,
    "ball5": 29.9, "ball6": 42.0, "gauss5": 0.1, "gauss6": 15.3, "ellip5": 0.1, "ellipN5": 0.0,
}

# 档2案例的 t1f,t2f (PCA环终点, 供容错带对称性展示)
TAUS_FIN = {
    "ball0": ((0.0000, 1.2793), (0.0000, 1.0000)),
    "ball1": ((-0.0843, 1.6345), (-0.2413, 2.5950)),
    "ball4": ((0.3792, 2.9749), (0.0760, 1.0551)),
    "gauss4": ((-0.4929, 3.4076), (0.3959, 4.3928)),
    "ellip1": ((0.2483, 2.7106), (-0.0420, 1.7530)),
    "ellipN0": ((0.3374, 1.4085), (0.1409, 3.0429)),
    "ellipN1": ((0.0000, 1.2793), (0.0000, 1.0000)),
    "ball5": ((-0.066, 1.557), (0.042, 1.006)),
    "gauss6": ((-0.308, 1.122), (0.457, 1.206)),
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


# ---------- 1. 新数据集补测 PCA 环 ----------
print("=== 新增数据集 PCA 环 (max_iter=40) ===")
for tag, seed in (("ball", 7), ("ball", 8), ("gauss", 7), ("gauss", 8),
                  ("ellip", 6), ("ellip", 7), ("ellipN", 6), ("ellipN", 7)):
    name = f"{tag}{seed}"
    pts = make_set(tag, seed)
    r30, r45, a, axis = vp.prepare(pts)
    t1f, t2f, F1f, F2f = m.refine_moduli_by_axis(pts, t10, t20, axis,
                                                 max_iter=40, verbose=False)
    d = F1f - F2f
    dn = d / np.linalg.norm(d)
    th = vp.ang_line(dn, axis)
    THETA_PCA[name] = th
    TAUS_FIN[name] = ((t1f.real, t1f.imag), (t2f.real, t2f.imag))
    print(f"{name}: θ_PCA = {th:.1f}°  t1f=({t1f.real:.3f},{t1f.imag:.3f}) "
          f"t2f=({t2f.real:.3f},{t2f.imag:.3f})")

# ---------- 2. 分类 ----------
names = list(THETA_PCA.keys())
tier1 = [n for n in names if THETA_PCA[n] < 0.5]
mid = [n for n in names if 0.5 <= THETA_PCA[n] <= 30.0]
tier2 = [n for n in names if THETA_PCA[n] > 30.0]
print()
print(f"档1 成立 (<0.5°): {len(tier1)} 个  {sorted(tier1)}")
print(f"容错带 (0.5~30°): {len(mid)} 个  {sorted(mid)}")
print(f"档2 准入 (>30°):  {len(tier2)} 个  {sorted(tier2)}")

# ---------- 3. 档2 跑概率环 500 次 ----------
print()
print("=" * 110)
print("档2: 概率环 500 次, 后100次角度范围≤3° 判定自收敛")
print("=" * 110)
results = {}
for name in sorted(tier2):
    tag = name.rstrip("0123456789")
    seed = int(name[len(tag):])
    pts = make_set(tag, seed)
    r30, r45, a, axis = vp.prepare(pts)
    hist = vf.prob_loop(pts, r30, r45, a, t10, t20, n_iter=500)
    angs = [vp.ang_line(dn, axis) if dn is not None else float("nan")
            for _, _, _, dn in hist]
    tail = angs[-4:]
    spread = max(tail) - min(tail)
    settled = spread <= 3.0
    fixed = sum(tail) / len(tail) if settled else float("nan")
    _, tf1, tf2, _ = hist[-1]
    d12 = abs(tf1 - tf2)
    D1 = abs(tf1 - 1j) + abs(tf2 - 1j)
    results[name] = (settled, fixed, spread)
    print(f"{name:8s} θ_PCA={THETA_PCA[name]:6.1f}°  概率环: "
          f"θ0={angs[0]:5.1f}° θ500={angs[-1]:5.1f}°  尾100=[{min(tail):.1f},{max(tail):.1f}]  "
          f"{'自收敛' if settled else '游走'}{' ' if settled else ''}"
          f"{f' 固定角={fixed:.1f}°' if settled else ''}  "
          f"|t1-t2|={d12:.3f} D1={D1:.3f}")

# ---------- 4. 容错带: 对称性数据 ----------
print()
print("=== 容错带 (0.5~30°): t1,t2 对称性 (PCA环终点) ===")
for name in sorted(mid):
    if name not in TAUS_FIN:
        continue
    (r1, i1), (r2, i2) = TAUS_FIN[name]
    t1 = complex(r1, i1)
    t2 = complex(r2, i2)
    print(f"{name:8s} θ={THETA_PCA[name]:6.1f}°  |t1-t2|={abs(t1-t2):6.3f}  "
          f"D1={abs(t1-1j)+abs(t2-1j):6.3f}  t1=({r1:.3f},{i1:.3f}) t2=({r2:.3f},{i2:.3f})")

# ---------- 5. 汇总 ----------
n_pass = len(tier1) + sum(1 for v in results.values() if v[0])
print()
print(f"汇总: 档1={len(tier1)} 档2通过={sum(1 for v in results.values() if v[0])}/"
      f"{len(results)}  容错带={len(mid)} (未进档2)")
print("DONE")
