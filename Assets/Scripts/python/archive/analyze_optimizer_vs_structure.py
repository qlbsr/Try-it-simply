# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
因果分解: |Δ|→0 的结果究竟是"优化器(LM/NM)的作用"还是"椭圆曲线结构的作用"?

判别实验:
  A. 纯几何控制: 随机方向 v, d0, d (不经任何椭圆曲线计算)
     → 若 V 形漏斗 / bisector 零点 / corr(|Δ|,|d·(v-d0)|) 在随机方向下同样成立,
       则 |Δ| 的"几何形态"与曲线无关 (方向空间几何事实)
  B. 结构可达域网格扫描: 不用优化器, 在 (t1,t2) 基本域上穷举网格计算 d(t1,t2)
     → 网格最优 |Δ| 若 ≈ 0, 则结果是"结构提供的", 优化器只是执行者
     → 网格最优 angle(d,v) 即结构可达域下限 (与优化器无关)
     → 盆地占比: |Δ|≤10° 的格点比例 (漏斗有多宽)
  C. 预算对比: NM vs 纯随机采样 (结构相同, 引导不同) → 优化器的边际贡献
     + 随机最优 → NM 抛光: NM 是"好起点下的好抛光器"还是"全局搜索器"?
  D. LM 步从网格最优出发: LM 是否还能再降? (分段常数平台 → 不能)
"""
import math
import time

import numpy as np
import scipy.optimize as opt

import data_driven_axis as dd
import n2sjy2 as n2

BOX = [(-0.5, 0.5), (0.5, 3.0), (-0.5, 0.5), (0.5, 3.0)]


def angle(u, v):
    return math.degrees(math.acos(np.clip(np.dot(u, v), -1, 1)))


def setup(pts):
    rp = n2.compute_rp(pts)
    d2 = math.asin(math.cos(math.radians(30)) / math.pi)
    a = rp * (1 + math.sin(d2))
    _, r45, r30, _ = n2.yzqx(pts, rp)
    v = n2.pca_axis(pts)
    t1t, t2t = n2.compute_taus()
    pt0, p10, p20, sig0 = dd.fast_probs(r30, r45, t1t, t2t, a)
    F10, F20 = n2.extract_foci(pts, p10, sig0[1], p20, sig0[2])
    d0 = np.array(F10, float) - np.array(F20, float)
    n0 = np.linalg.norm(d0)
    if n0 < 1e-12:
        d0 = np.array([1.0, 0.0, 0.0])
    else:
        d0 = d0 / n0
    return r30, r45, a, v, d0


def direction_at(x, pts, r30, r45, a):
    """(t1,t2) → 概率 → 焦点方向 d (椭圆曲线结构的映射)"""
    t1 = complex(x[0], x[1])
    t2 = complex(x[2], x[3])
    pt, p1, p2, sig = dd.fast_probs(r30, r45, t1, t2, a)
    F1, F2 = n2.extract_foci(pts, p1, sig[1], p2, sig[2])
    d = np.array(F1, float) - np.array(F2, float)
    n = np.linalg.norm(d)
    return d / n if n > 1e-9 else None


def delta_of(d, v, d0):
    return abs(angle(d, v) - angle(d, d0))


def nm_minimize(pts, objective, x0, max_evals=400):
    r30, r45, a, v, d0 = setup(pts)
    evals = [0]

    def cost(x):
        evals[0] += 1
        return objective(x, pts, r30, r45, a, v, d0)

    x0c = np.clip(x0, [b[0] for b in BOX], [b[1] for b in BOX])
    res = opt.minimize(cost, x0c, method="Nelder-Mead", bounds=BOX,
                       options={"maxiter": max_evals, "maxfev": max_evals * 2})
    return res.x, evals[0]


def obj_angle_v(x, pts, r30, r45, a, v, d0):
    d = direction_at(x, pts, r30, r45, a)
    return 1e6 if d is None else angle(d, v)


def obj_delta(x, pts, r30, r45, a, v, d0):
    d = direction_at(x, pts, r30, r45, a)
    return 1e6 if d is None else delta_of(d, v, d0)


# ============ Part A: 纯几何控制 ============
def part_a(n_trials=300, n_samples=3000, seed=42):
    rng = np.random.default_rng(seed)
    min_deltas, bis_deltas, corrs = [], [], []
    for _ in range(n_trials):
        v = rng.standard_normal(3)
        v /= np.linalg.norm(v)
        d0 = rng.standard_normal(3)
        d0 /= np.linalg.norm(d0)
        D = rng.standard_normal((n_samples, 3))
        D /= np.linalg.norm(D, axis=1, keepdims=True)
        deltas = np.array([delta_of(d, v, d0) for d in D])
        bis = v + d0
        bis /= np.linalg.norm(bis)
        bis_deltas.append(delta_of(bis, v, d0))
        proj = np.abs(D @ (v - d0))
        if proj.std() > 0:
            corrs.append(np.corrcoef(deltas, proj)[0, 1])
        min_deltas.append(deltas.min())
    print("=" * 70)
    print("Part A  纯几何控制 (随机方向, 无任何椭圆曲线计算)")
    print("=" * 70)
    print(f"  随机 d 采样 {n_samples}/轮 × {n_trials} 轮:")
    print(f"    min|Δ| 分布: 中位 {np.median(min_deltas):.3f}° 最大 {np.max(min_deltas):.3f}°"
          f" (≤0.5°占比 {(np.array(min_deltas)<=0.5).mean()*100:.0f}%)")
    print(f"    bisector 处 |Δ|: 全部 ≤ {max(bis_deltas):.2e}° (解析零点)")
    print(f"    corr(|Δ|,|d·(v-d0)|): 中位 {np.median(corrs):.3f}")
    print("  → |Δ| 漏斗形态是方向空间几何事实, 与曲线无关; 零点=等角大圆∩球面")
    return min_deltas, bis_deltas, corrs


# ============ Part B: 结构可达域网格扫描 (不用优化器) ============
def part_b(pts, name, grid_n=10):
    r30, r45, a, v, d0 = setup(pts)
    re = np.linspace(-0.5, 0.5, grid_n)
    im = np.linspace(0.5, 3.0, grid_n)
    best_delta, best_delta_x, best_ang, best_ang_x = 1e9, None, 1e9, None
    n_valid, n_basin = 0, 0
    t0 = time.time()
    for r1 in re:
        for i1 in im:
            for r2 in re:
                for i2 in im:
                    d = direction_at([r1, i1, r2, i2], pts, r30, r45, a)
                    if d is None:
                        continue
                    n_valid += 1
                    dl = delta_of(d, v, d0)
                    if dl < best_delta:
                        best_delta, best_delta_x = dl, (r1, i1, r2, i2)
                    if dl <= 10.0:
                        n_basin += 1
                    ag = angle(d, v)
                    if ag < best_ang:
                        best_ang, best_ang_x = ag, (r1, i1, r2, i2)
    print(f"  {name}: 网格 {grid_n}^4={grid_n**4} 点 (有效 {n_valid}) | "
          f"min|Δ|={best_delta:.2f}° | min∠(d,v)={best_ang:.2f}° @ {tuple(round(q,2) for q in best_ang_x)}"
          f" | |Δ|≤10° 盆地占比 {n_basin/max(n_valid,1)*100:.0f}% | {time.time()-t0:.0f}s")
    return best_delta, best_ang, n_basin / max(n_valid, 1)


# ============ Part C: 预算对比 ============
def part_c(pts, name, n_random=4096, seed=7):
    r30, r45, a, v, d0 = setup(pts)
    t1t, t2t = n2.compute_taus()
    x0 = [t1t.real, t1t.imag, t2t.real, t2t.imag]

    # NM 最小化 |Δ| (单起点, 理论起点)
    xd, ev_d = nm_minimize(pts, obj_delta, x0)
    dd_nm = direction_at(xd, pts, r30, r45, a)
    nm_delta = delta_of(dd_nm, v, d0) if dd_nm is not None else float('nan')
    nm_ang = angle(dd_nm, v) if dd_nm is not None else float('nan')

    # NM 最小化 angle(d,v) (单起点)
    xa, ev_a = nm_minimize(pts, obj_angle_v, x0)
    da = direction_at(xa, pts, r30, r45, a)
    nm_ang_v = angle(da, v) if da is not None else float('nan')

    # 纯随机采样 (同预算)
    rng = np.random.default_rng(seed)
    xs = np.column_stack([rng.uniform(-0.5, 0.5, n_random),
                          rng.uniform(0.5, 3.0, n_random),
                          rng.uniform(-0.5, 0.5, n_random),
                          rng.uniform(0.5, 3.0, n_random)])
    rand_delta, rand_ang, rand_best_x = 1e9, 1e9, None
    for x in xs:
        d = direction_at(x, pts, r30, r45, a)
        if d is None:
            continue
        dl = delta_of(d, v, d0)
        ag = angle(d, v)
        if dl < rand_delta:
            rand_delta = dl
        if ag < rand_ang:
            rand_ang, rand_best_x = ag, x

    # 随机最优 → NM 抛光 (∠v)
    polish = float('nan')
    if rand_best_x is not None:
        xp, _ = nm_minimize(pts, obj_angle_v, rand_best_x, max_evals=200)
        dp = direction_at(xp, pts, r30, r45, a)
        if dp is not None:
            polish = angle(dp, v)

    print(f"  {name}: NM(|Δ|)={nm_delta:.1f}°({ev_d}次,∠(d,v)={nm_ang:.0f}°) "
          f"NM(∠v)={nm_ang_v:.1f}°({ev_a}次) | "
          f"随机{n_random}点: min|Δ|={rand_delta:.2f}° min∠(d,v)={rand_ang:.2f}° "
          f"→ 随机最优+NM抛光 ∠(d,v)={polish:.1f}°")
    return nm_delta, nm_ang_v, rand_delta, rand_ang, polish


# ============ Part D: LM 从网格最优出发 ============
def part_d(pts, name, n_inner=60):
    r30, r45, a, v, d0 = setup(pts)
    x = np.array([0.3571428571428571, 0.5, -0.07142857142857145, 0.8571428571428572], float) \
        if name == "ellip0" else np.array([0.0714285714285714, 1.9285714285714286, 0.2142857142857142, 0.5], float)
    x = np.clip(x, [b[0] for b in BOX], [b[1] for b in BOX])
    lam = 1e-3
    moved = 0.0
    for _ in range(n_inner):
        xp, lam, F1, F2 = dd.lm_step(x, pts, v, r30, r45, a, lam=lam)
        step = np.max(np.abs(xp - x))
        if step < 1e-9:
            break
        x = xp
        moved = max(moved, step)
    d = direction_at(x, pts, r30, r45, a)
    print(f"  {name}: LM {n_inner} 步后 |Δ|={delta_of(d, v, d0) if d is not None else float('nan'):.2f}° "
          f"∠(d,v)={angle(d, v) if d is not None else float('nan'):.2f}° (最大步长 {moved:.2e})")
    return x


def main():
    t_start = time.time()
    part_a()
    print()
    datasets = {"cube0": dd.make_cube(10), "ellip0": dd.make_ellip(0),
                "ball0": dd.make_ball(0), "ball1": dd.make_ball(1)}
    print("=" * 70)
    print("Part B  结构可达域网格扫描 (不用优化器, 穷举 (t1,t2) 基本域)")
    print("=" * 70)
    for name, pts in datasets.items():
        part_b(pts, name)
    print()
    print("=" * 70)
    print("Part C  预算对比: NM vs 纯随机采样 (结构相同, 引导不同)")
    print("=" * 70)
    for name, pts in datasets.items():
        try:
            part_c(pts, name)
        except Exception as e:
            print(f"  {name}: 跳过 ({e})")
    print()
    print("=" * 70)
    print("Part D  LM 从近似网格最优出发 (优化器类型的影响)")
    print("=" * 70)
    for name, pts in datasets.items():
        try:
            part_d(pts, name)
        except Exception as e:
            print(f"  {name}: 跳过 ({e})")
    print()
    print(f"TOTAL {time.time()-t_start:.0f}s  DONE")


if __name__ == "__main__":
    main()
