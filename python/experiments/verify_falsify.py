# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
按用户放宽后的证伪标准重新判定:
  证伪A: 初始角θ(theory)>30° 且 最终角θ_final>30°, 但 t1,t2 非常接近 (0,1) (D1 小、对称)
  证伪B: 最终角很小(<5°), 但 t1,t2 远离 (0,1) / 不对称  (反向: 小角≠贴近(0,1))
补: ellipN 系列的初始角, 以及 6 个新数据集的完整扫描
"""
import numpy as np

import nsjy_algorithms as m
import verify_probability_correction as vp

t10, t20 = m.compute_taus()

# ---------- 1. 补算 ellipN 系列的初始角 ----------
print("=== 补算 ellipN 初始角 θ(theory) ===")
for name in ("ellipN0", "ellipN1", "ellipN2", "ellipN3", "ellipN4"):
    pts = vp.build(name)
    r30, r45, a, axis = vp.prepare(pts)
    _, _, _, dn = vp.refit(pts, r30, r45, a, t10, t20)
    print(f"{name}: θ_theory = {vp.ang_line(dn, axis):.1f}°")
vp.TH_THEORY["ellipN0"] = None  # placeholder; 下面在扫描里一并算

# ---------- 2. 按证伪标准分类 20 个已有案例 ----------
print()
print("=== 证伪A: θ_init>30 且 θ_final>30 且 D1<0.6 (贴近(0,1)却大角) ===")
fa = []
for name, ((r1, i1), (r2, i2), thf) in vp.BATCH.items():
    th0 = vp.TH_THEORY.get(name)
    if th0 is None:
        continue
    t1 = complex(r1, i1)
    t2 = complex(r2, i2)
    D1 = abs(t1 - 1j) + abs(t2 - 1j)
    d12 = abs(t1 - t2)
    if th0 > 30.0 and thf > 30.0 and D1 < 0.6:
        fa.append((name, th0, thf, d12, D1))
        print(f"  {name:8s} θ_init={th0:6.1f}°  θ_final={thf:6.1f}°  "
              f"|t1-t2|={d12:.3f}  D1={D1:.3f}   t1=({r1:.3f},{i1:.3f}) t2=({r2:.3f},{i2:.3f})")
if not fa:
    print("  (无)")

print()
print("=== 证伪B: θ_final<5° 但 D1>1.0 (小角却远离(0,1)) ===")
for name, ((r1, i1), (r2, i2), thf) in vp.BATCH.items():
    t1 = complex(r1, i1)
    t2 = complex(r2, i2)
    D1 = abs(t1 - 1j) + abs(t2 - 1j)
    if thf < 5.0 and D1 > 1.0:
        print(f"  {name:8s} θ_final={thf:5.2f}°  D1={D1:.3f}  |t1-t2|={abs(t1-t2):.3f}")

# ---------- 3. 新数据集扫描 (6 个) ----------
print()
print("=== 新数据集扫描 (PCA 环 max_iter=40) ===")
# build() 只支持 seed 0-4; 手动构造新 seed
def build_new(tag, seed, n=200):
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

for tag, seed in (("ball", 5), ("ball", 6), ("gauss", 5), ("gauss", 6),
                  ("ellip", 5), ("ellipN", 5)):
    name = f"{tag}{seed}"
    pts = build_new(tag, seed)
    r30, r45, a, axis = vp.prepare(pts)
    _, _, _, dn0 = vp.refit(pts, r30, r45, a, t10, t20)
    th_init = vp.ang_line(dn0, axis)
    t1f, t2f, F1f, F2f = m.refine_moduli_by_axis(pts, t10, t20, axis,
                                                 max_iter=40, verbose=False)
    d = F1f - F2f
    dn = d / np.linalg.norm(d)
    th_fin = vp.ang_line(dn, axis)
    D1 = abs(t1f - 1j) + abs(t2f - 1j)
    d12 = abs(t1f - t2f)
    tagA = " <== 证伪A" if (th_init > 30 and th_fin > 30 and D1 < 0.6) else ""
    tagB = " <== 证伪B" if (th_fin < 5 and D1 > 1.0) else ""
    print(f"{name:8s} θ_init={th_init:6.1f}°  θ_final={th_fin:6.1f}°  "
          f"D1={D1:.3f}  |t1-t2|={d12:.3f}  t1f=({t1f.real:.3f},{t1f.imag:.3f}) "
          f"t2f=({t2f.real:.3f},{t2f.imag:.3f}){tagA}{tagB}")

print()
print("DONE")
