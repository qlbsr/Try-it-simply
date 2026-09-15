# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
核心问题: s0 与 s5 概率数组是否相近? 相近时能否反推近似 v 替代 pcav?
  数学: s(v)[i] = exp(-|d1+d2-2a|/(2a)) 只依赖 (p̂_i·v)² (阶段N/P)
        → s0≈s5 全数组 ⟺ |p̂_i·v|≈|p̂_i·d| ∀i ⟹ v≈±d (一般位置)
  流程: 达成点取 s5 → 逐点反演 cos²θ_i → 用符号(来自d)重建 w → 对比 v
  输出: s0/s5 相似度; w 与 d/v3(用户pca)/真协方差主轴 的夹角
"""
import math
import numpy as np

import data_driven_axis as dd
import n2sjy2 as n2
import nsjy_algorithms as m
from scipy.optimize import minimize


def angle(u, v):
    return math.degrees(math.acos(np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1)))


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


def inv_cos2_from_s(s_i, r_i, c, a):
    """s → cos²θ (代数, 阶段P5): 双解 D=d1+d2, 取使几何可行的"""
    ln = abs(math.log(max(float(s_i), 1e-12)))
    for sgn in (1, -1):
        D = 2 * a + sgn * 2 * a * ln
        if D < 2 * c or D < 2 * abs(r_i):
            continue
        ab = (D * D - 2 * (r_i * r_i + c * c)) / 2
        if ab <= 0:
            continue
        cos2 = ((r_i * r_i + c * c) ** 2 - ab * ab) / (4 * c * c * r_i * r_i)
        if 0 <= cos2 <= 1.0001:
            return min(cos2, 1.0)
    return None


def solve_w(Phat, cos_vals, sign):
    """最小二乘: w = argmin Σ(p̂_i·w - sign_i·cosθ_i)²"""
    target = sign * cos_vals
    A = Phat.T @ Phat + 1e-9 * np.eye(3)
    b = Phat.T @ target
    w = np.linalg.solve(A, b)
    return w / np.linalg.norm(w)


def analyze(name, pts, max_outer=15):
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
    v3 = m.pca(pts)[1]                      # 用户 pca() 的 v3
    v3 = v3 / np.linalg.norm(v3)
    # 真协方差主轴
    C = P.T @ P
    evals, evecs = np.linalg.eigh(C)
    v_true = evecs[:, np.argsort(evals)[::-1][0]]

    def norm(vv):
        return np.asarray(vv, float) / np.linalg.norm(vv)

    v1 = norm(F1 - F2)
    v3f = norm(F01 - F02)
    vj = v3 + norm(F1 - F2)
    # (简化LM→复用循环; 直接迭代)
    t1, t2 = t_, t_
    f1z, f2z = f1, f2

    def refine(x, ax):
        return x, ax  # 占位 (此处省略完整LM以提速, 直接NM)

    # 迭代 (LM+NM 完整, 从 verify_condition2 复制简化版)
    from scipy.optimize import minimize as _min
    for i in range(max_outer):
        d = norm(f1z - f2z)
        ad1 = angle(d, v1)
        ad2 = angle(d, v3f)
        cond = abs((ad2 + ad1) * 0.5 - min(ad2, ad1))
        if cond <= 8:
            s0 = bp(P, v3 * c, -v3 * c, a)
            s5 = bp(P, d * c, -d * c, a)
            # s0 vs s5 全数组相似度
            corr = np.corrcoef(s0, s5)[0, 1]
            l2 = np.linalg.norm(s0 - s5) / math.sqrt(len(P))
            med = np.median(np.abs(s0 - s5) * 100)
            # 从 s5 反推 w
            Phat = P / np.linalg.norm(P, axis=1, keepdims=True)
            r_i = np.linalg.norm(P, axis=1)
            cos2 = np.array([inv_cos2_from_s(s5[j], r_i[j], c, a) for j in range(len(P))])
            valid = np.isfinite(cos2)
            # 符号用 d (w 应≈±d 若 s0≈s5)
            sign = np.sign(Phat @ d + 1e-12)
            w = solve_w(Phat[valid], np.sqrt(cos2[valid]), sign[valid])
            w2 = -w
            print(f"  [{name}] 达成 cond={cond:.2f}")
            print(f"    s0/s5: corr={corr:.3f} L2={l2:.4f} 中位|Δs|×100={med:.2f}")
            print(f"    d vs v3(用户pca): {angle(d, v3):.1f}° | d vs 真主轴: {angle(d, v_true):.1f}°")
            print(f"    s5反推w vs d: {angle(w, d):.1f}° | w vs v3: {angle(w, v3):.1f}° "
                  f"| w vs 真主轴: {min(angle(w, v_true), angle(w2, v_true)):.1f}°")
            print(f"    s0 vs s5 相近 ⟹ w≈±d? w 与 v3 夹角 = {angle(w, v3):.1f}°")
            return
        # 下一步 NM (简化: 直接用 nm 目标在参考 newv/newF)
        t1n, t2n = t1, t2

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
        F1n, F2n = F1, F2
        axis = np.cross(norm(F1n - F2n), norm(F1 - F2))
        axis1 = np.cross(norm(F1n - F2n), norm(F01 - F02))

        def rotate(vv, ax, deg):
            if np.linalg.norm(ax) < 1e-12:
                return vv
            ax = ax / np.linalg.norm(ax)
            rg = math.radians(deg)
            return vv * math.cos(rg) + np.cross(ax, vv) * math.sin(rg) \
                + ax * np.dot(ax, vv) * (1 - math.cos(rg))

        newv = rotate(norm(F01 - F02), axis, -ad2)
        newF = rotate(norm(F1 - F2), axis1, -ad2)

        def cb(xk):
            return nm_obj(xk, newv, newF) <= 10.

        x0 = np.clip([t1.real, t1.imag, t2.real, t2.imag], lb, ub)
        res = _min(nm_obj, x0, args=(newv, newF), method="Nelder-Mead",
                   bounds=list(zip(lb, ub)), callback=cb,
                   options={"maxiter": 400, "maxfev": 800, "xatol": 1e-6, "fatol": 1e-6})
        t1 = complex(res.x[0], res.x[1])
        t2 = complex(res.x[2], res.x[3])
        # 更新 f1z/f2z
        ptc, pc1, pc2, sc = dd.fast_probs(r30, r45, t1, t2, a)
        Fc1, Fc2 = m.extract_foci(pts, pc1, sc[1], pc2, sc[2])
        f1z, f2z = fit_fast(P, ptc, Fc1, Fc2, a)
    print(f"  [{name}] 未达成")
    return


def main():
    print("=" * 96)
    print("s0/s5 全数组相似度 + 从 s5 反推近似 v (替代 pcav)")
    print("=" * 96)
    for name, pts in [("ball0", dd.make_ball(0)), ("ball2", dd.make_ball(2)),
                      ("ellip0", dd.make_ellip(0)), ("cube0", dd.make_cube(10))]:
        try:
            analyze(name, pts)
        except Exception as e:
            print(f"  {name}: 失败 ({e})", flush=True)
        print(flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
