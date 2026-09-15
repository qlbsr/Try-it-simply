# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
10 夹角的关系/相关性分析 (n2sjy2.cs)
方向向量:
  d     = f1-f2       理论taus拟合焦点(迭代更新)
  Fd    = F1-F2       理论taus提取焦点(固定)
  F01d  = F01-F02     (0,1)提取焦点(固定)
  fd01  = f01-f02     (0,1)拟合焦点(固定)
  Fnd   = F1_new-F2_new  LM精化提取(每轮)
  fnd   = f_1new-f_2new  LM后拟合(每轮)
  ffd   = f1f-f2f     初始理论拟合(固定)
  v     = PCA主轴
角度:
  ang1 = ∠(d, v)     ang2 = ∠(d, Fd)     ang3 = ∠(d, F01d)
  ang4 = ∠(Fnd, Fd)  ang5 = ∠(Fnd, F01d) ang6 = ∠(fnd, ffd)
  ang7 = ∠(fnd, fd01) ang8 = ∠(fnd, v)   ang9 = ∠(Fnd, v)
  用户代号: angleDeg1=∠(d,v)=ang1... (见代码映射)
分析: 迭代全程收集 → 相关矩阵 → 冗余/独立度 → 与收敛(cond#2)的关系
"""
import math
import numpy as np

import data_driven_axis as dd
import n2sjy2 as n2
import nsjy_algorithms as m
from scipy.optimize import minimize


def angle(u, vv):
    return math.degrees(math.acos(np.clip(np.dot(u, vv) / (np.linalg.norm(u) * np.linalg.norm(vv)), -1, 1)))


def bp(P, F1, F2, a):
    d1 = np.linalg.norm(P - F1, axis=1)
    d2 = np.linalg.norm(P - F2, axis=1)
    delta = d1 + d2 - 2 * a
    return np.exp(-np.abs(delta) / (2 * a))


def fit_fast(P, prob, F1, F2, a, iters=300, lr=0.0001):
    F1 = np.array(F1, float).copy()
    F2 = np.array(F2, float).copy()
    for _ in range(iters):
        d1 = np.linalg.norm(P - F1, axis=1)
        d2 = np.linalg.norm(P - F2, axis=1)
        sd1 = np.maximum(d1, 1e-3)
        sd2 = np.maximum(d2, 1e-3)
        delta = d1 + d2 - 2 * a
        fp = np.exp(-np.abs(delta) / (2 * a))
        diff = fp - prob
        loss = float(np.sum(diff * diff))
        if loss < 1e-12:
            break
        sign = np.where(delta >= 0, 1., -1.)
        coef = diff * fp * sign / (2 * a)
        g1 = coef[:, None] * (P - F1) / sd1[:, None]
        g2 = coef[:, None] * (P - F2) / sd2[:, None]
        grad1 = g1.sum(0)
        grad2 = g2.sum(0)
        sep = F1 - F2
        grad1 += 2 * sep
        grad2 -= 2 * sep
        n1 = np.linalg.norm(grad1)
        n2 = np.linalg.norm(grad2)
        if n1 > 5:
            grad1 *= 5 / n1
        if n2 > 5:
            grad2 *= 5 / n2
        F1 -= lr * grad1
        F2 -= lr * grad2
        if not (np.all(np.isfinite(F1)) and np.all(np.isfinite(F2))):
            break
    return F1, F2


def collect_angles(name, pts, max_outer=15):
    """迭代全程收集 10 角 (每轮) + 收敛标记"""
    pts = [np.array(p, float) for p in pts]
    P = np.array(pts, float)
    rp = n2.compute_rp(pts)
    d2 = math.asin(math.cos(math.radians(30)) / math.pi)
    a = rp * (1 + math.sin(d2))
    e3 = math.cos(d2) * 2 / (1 + math.sin(d2))
    c = rp * math.cos(d2) * e3
    _, r45, r30, _ = n2.yzqx(pts, rp)
    t1t, t2t = n2.compute_taus()
    t_ = complex(0, 1)
    pt, p1, p2, sig = dd.fast_probs(r30, r45, t1t, t2t, a)
    F1, F2 = m.extract_foci(pts, p1, sig[1], p2, sig[2])
    f1, f2 = fit_fast(P, pt, F1, F2, a)
    pt01, p101, p201, sig01 = dd.fast_probs(r30, r45, t_, t_, a)
    F01, F02 = m.extract_foci(pts, p101, sig01[1], p201, sig01[2])
    f01, f02 = fit_fast(P, pt01, F01, F02, a)
    v = m.pca(pts)[1]
    v = v / np.linalg.norm(v)

    def norm(vv):
        return np.asarray(vv, float) / np.linalg.norm(vv)

    Fd = norm(F1 - F2)        # 理论提取(固定)
    F01d = norm(F01 - F02)    # (0,1)提取(固定)
    fd01 = norm(f01 - f02)    # (0,1)拟合(固定)
    ffd = norm(f1 - f2)       # 初始理论拟合(固定, f1f-f2f)
    # f1f/f2f 在代码里 = 初始 f1/f2 (fit 后立即保存)

    vj = v + Fd
    t1, t2 = t_, t_
    f1z, f2z = f1, f2
    rows = []
    for i in range(max_outer):
        d = norm(f1z - f2z)         # 当前 (f1-f2)
        # LM 一步得到 F1_new/F2_new 与 f_1new/f_2new (简化: 用当前t1,t2直接提取+拟合模拟)
        t1n, t2n = t1, t2
        ptn, p1n, p2n, sgn = dd.fast_probs(r30, r45, t1n, t2n, a)
        F1n, F2n = m.extract_foci(pts, p1n, sgn[1], p2n, sgn[2])
        f1n, f2n = fit_fast(P, ptn, F1n, F2n, a)
        Fnd = norm(F1n - F2n)
        fnd = norm(f1n - f2n)
        # 10 个角 (按用户定义)
        a1 = angle(d, v)          # angleDeg  (f1-f2 vs v)
        a2 = angle(d, Fd)         # angleDeg1 (f1-f2 vs F1-F2)
        a3 = angle(d, F01d)       # angleDeg2 (f1-f2 vs F01-F02)
        a4 = angle(Fnd, Fd)       # angleDeg3 (F1_new-F2_new vs F1-F2)
        a5 = angle(Fnd, F01d)     # angleDeg4 (F1_new-F2_new vs F01-F02)
        a6 = angle(fnd, ffd)      # angleDeg5 (f_1new-f_2new vs f1f-f2f)
        a7 = angle(fnd, fd01)     # angleDeg6 (f_1new-f_2new vs f01-f02)
        a8 = angle(fnd, v)        # angleDeg9 (f_1new-f_2new vs v)
        a9 = angle(Fnd, v)        # angleDeg10(F1_new-F2_new vs v)
        # 达成条件 (用户当前用 angleDeg5/6 → a6,a7)
        cond_56 = abs((a6 + a7) * 0.5 - min(a6, a7))
        ok = (a6 >= 16 and a7 >= 16 and cond_56 <= 8)
        rows.append([a1, a2, a3, a4, a5, a6, a7, a8, a9, float(ok)])
        if ok:
            break
        if i >= max_outer - 1:
            break
        # NM 一步推进 (用 fnd 作参考, 与用户一致: RefineTausWithNM(t1_new,t2_new,...,f_1new,f_2new))
        def nm_obj(x, pcav, f1f2v):
            tt1 = complex(x[0], x[1])
            tt2 = complex(x[2], x[3])
            ptc, pc1, pc2, sc = dd.fast_probs(r30, r45, tt1, tt2, a)
            Fc1, Fc2 = m.extract_foci(pts, pc1, sc[1], pc2, sc[2])
            fc1, fc2 = fit_fast(P, ptc, Fc1, Fc2, a)
            dd_ = fc1 - fc2
            if np.dot(dd_, dd_) < 1e-12:
                return 0.
            dd_ = dd_ / np.linalg.norm(dd_)
            return abs(angle(dd_, pcav) - angle(dd_, f1f2v))

        lb = np.array([-0.5, 0.5, -0.5, 0.5])
        ub = np.array([0.5, 1.5, 0.5, 1.5])

        def cb(xk):
            return nm_obj(xk, fnd, ffd) <= 10.

        x0 = np.clip([t1.real, t1.imag, t2.real, t2.imag], lb, ub)
        try:
            rr = minimize(nm_obj, x0, args=(fnd, ffd), method="Nelder-Mead",
                          bounds=list(zip(lb, ub)), callback=cb,
                          options={"maxiter": 400, "maxfev": 800, "xatol": 1e-6, "fatol": 1e-6})
            t1 = complex(rr.x[0], rr.x[1])
            t2 = complex(rr.x[2], rr.x[3])
        except Exception:
            pass
        ptn, p1n, p2n, sgn = dd.fast_probs(r30, r45, t1, t2, a)
        F1n, F2n = m.extract_foci(pts, p1n, sgn[1], p2n, sgn[2])
        f1z, f2z = fit_fast(P, ptn, F1n, F2n, a)
    return np.array(rows) if rows else None


def main():
    datasets = {}
    for seed in range(4):
        datasets[f"ball{seed}"] = dd.make_ball(seed)
    for seed in range(3):
        datasets[f"ellip{seed}"] = dd.make_ellip(seed)
    datasets["cube0"] = dd.make_cube(10)

    names = ["ang(d,v)", "ang(d,Fd)", "ang(d,F01d)", "ang(Fnd,Fd)", "ang(Fnd,F01d)",
             "ang(fnd,ffd)", "ang(fnd,fd01)", "ang(fnd,v)", "ang(Fnd,v)"]
    allrows = []
    for name, pts in datasets.items():
        try:
            rows = collect_angles(name, pts)
        except Exception as e:
            print(f"  {name}: 失败 ({e})", flush=True)
            continue
        if rows is None:
            continue
        print(f"  {name}: {len(rows)} 轮, 达成={rows[-1][-1] == 1}", flush=True)
        allrows.append(rows)
    R = np.vstack(allrows)
    angs = R[:, :9]
    ok = R[:, 9] == 1
    print()
    print("=" * 100)
    print(f"共 {len(R)} 轮样本 (达成轮 {ok.sum()})")
    print("=" * 100)
    # 相关性矩阵
    C = np.corrcoef(angs.T)
    print("相关性矩阵 (|corr|):")
    hdr = "         " + "".join(f"{n[:6]:>8s}" for n in names)
    print(hdr)
    for i in range(9):
        row = "".join(f"{abs(C[i, j]):8.2f}" for j in range(9))
        print(f"  {names[i][:8]:6s}{row}")
    print()
    # 与达成 (ok) 的区分度: |ang_at_ok_mean - ang_not_mean| / pooled std (判别力)
    print("各角对达成/未达成的判别力 (|Δ均值| / 合并std):")
    if ok.sum() >= 3 and (~ok).sum() >= 3:
        for i, n in enumerate(names):
            g0 = angs[~ok, i]
            g1 = angs[ok, i]
            pooled = math.sqrt((g0.std() ** 2 + g1.std() ** 2) / 2)
            d_ = abs(g1.mean() - g0.mean()) / (pooled + 1e-9)
            print(f"  {n:14s}: 达成均值 {g1.mean():6.1f}° 未达成 {g0.mean():6.1f}° "
                  f"判别力 {d_:.2f}")
    print()
    print("注: 判别力高 → 该角(或组合)最能区分达成; 相关高 → 冗余")
    print("DONE")


if __name__ == "__main__":
    main()
