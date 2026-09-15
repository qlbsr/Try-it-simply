# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
主动调整策略: 喂给 LM 前, 把方向 d=(f1-f2) 预旋转到与参考 r=(F01-F02) 成 60°左右
  原理: LM 的 axis 若=d(自指)→一步对齐→活性死; 若预旋转到与 r 成 60°,
        LM 把 d 拉到 axis 后 ∠(d,r)≈60°, 仍有推动空间, 不一次到位
  实现:
    target_angle(axis, r) = 60°: axis = rotate(d, n=d×r, ±α) 使夹角=60°
    - θ=∠(d,r)>60: 朝 r 转 (θ-60)°   ; θ<60: 远离 r 转 (60-θ)°
    - θ≈60: 微扰随机方位 (绕 r 旋转任意角, 保持60°锥面) → 探索
  变体:
    A: 现状 axis=d (自指)
    B: axis=与 r 成60°(同平面最近)
    C: axis=与 r 成60° + 绕 r 随机方位 (探索)
"""
import math
import numpy as np

import data_driven_axis as dd
import n2sjy2 as n2
import nsjy_algorithms as m
from scipy.optimize import minimize


def angle(u, vv):
    return math.degrees(math.acos(np.clip(np.dot(u, vv) / (np.linalg.norm(u) * np.linalg.norm(vv)), -1, 1)))


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


def rotate(vv, ax, deg, rng=None):
    ax = np.asarray(ax, float)
    if np.linalg.norm(ax) < 1e-12:
        return vv.copy()
    ax = ax / np.linalg.norm(ax)
    rg = math.radians(deg)
    v = vv * math.cos(rg) + np.cross(ax, vv) * math.sin(rg) \
        + ax * np.dot(ax, vv) * (1 - math.cos(rg))
    return v / np.linalg.norm(v)


def make_axis_at_angle(d, r, target=60.0, mode="B", rng=None):
    """把 d 调整为与 r 夹角=target 的方向 (保持与 d 尽量近 / 或随机方位)"""
    d = d / np.linalg.norm(d)
    r = r / np.linalg.norm(r)
    th = angle(d, r)
    n = np.cross(d, r)
    if np.linalg.norm(n) < 1e-9:
        # d ∥ r 或 d ∥ -r: 用任意垂直轴
        tmp = np.array([1., 0, 0]) if abs(d[0]) < 0.9 else np.array([0., 1, 0])
        n = np.cross(d, tmp)
        n = n / np.linalg.norm(n)
    if mode == "B":
        # 同平面: 转到与 r 夹角=target, 最小旋转
        delta = th - target   # 需改变的角度(带符号, 朝r为负方向? 用直接构造更稳)
        # 直接构造: 在 d-r 平面内, d' = cosT*r + sinT*(r_perp 方向校正)
        # r_perp = normalize(d - cosθ*r) 指向 d 的垂直分量
        cosT = math.cos(math.radians(target))
        cosTh = np.dot(d, r)
        if abs(cosTh) < 0.9999:
            r_perp = d - cosTh * r
            r_perp = r_perp / np.linalg.norm(r_perp)
            # 解: d'·r=cosT, d' 在平面内: d' = cosT*r ± sinT*r_perp
            # 选与 d 点积大者 (最接近 d)
            d1 = cosT * r + math.sin(math.radians(target)) * r_perp
            d2 = cosT * r - math.sin(math.radians(target)) * r_perp
            axis = d1 if np.dot(d1, d) > np.dot(d2, d) else d2
            return axis / np.linalg.norm(axis)
        else:
            return r if cosTh > 0 else -r
    else:  # mode C: 60° 锥面上随机方位
        cosT = math.cos(math.radians(target))
        # 先取任意 60° 方向 (同平面最接近 d 的解), 再绕 r 随机旋转方位
        base = make_axis_at_angle(d, r, target, "B")
        # 绕 r 旋转 random 方位角 (0~2π), 保持与 r 夹角不变
        ang_az = rng.uniform(0, 2 * math.pi)
        # 分解: base = cosT*r + sinT*q, q ⟂ r; 旋转 q 绕 r
        q = base - cosT * r
        q = q / np.linalg.norm(q)
        # q 绕 r 转 ang_az
        q2 = rotate(q, r, math.degrees(ang_az))
        return (cosT * r + math.sin(math.radians(target)) * q2)


def refine_lm(pts, r30, r45, a, t10, t20, axis_v, n_iter=50):
    P = np.array(pts, float)
    axis = np.asarray(axis_v, float)
    axis = axis / np.linalg.norm(axis)
    x = np.array([t10.real, t10.imag, t20.real, t20.imag], float)

    def resid(xx):
        t1 = complex(xx[0], xx[1])
        t2 = complex(xx[2], xx[3])
        pt, p1, p2, sig = dd.fast_probs(r30, r45, t1, t2, a)
        F1c, F2c = m.extract_foci(pts, p1, sig[1], p2, sig[2])
        r = np.zeros(len(pts) + 9)
        d1 = np.linalg.norm(P - F1c, axis=1)
        d2 = np.linalg.norm(P - F2c, axis=1)
        delta = d1 + d2 - 2 * a
        pf = np.exp(-np.abs(delta) / (2 * a))
        r[:len(pts)] = (pt - pf) / math.sqrt(len(pts))
        dv = F1c - F2c
        if np.linalg.norm(dv) < 1e-12:
            dv = axis
        dv = dv / np.linalg.norm(dv)
        if np.dot(dv, axis) < 0:
            dv = -dv
        e = dv - axis
        r[len(pts)] = 1.0 * e[0]
        r[len(pts) + 1] = 1.0 * e[1]
        r[len(pts) + 2] = 1.0 * e[2]
        return r, F1c, F2c

    lam = 1e-3
    for it in range(n_iter):
        r, F1, F2 = resid(x)
        cost = float(np.sum(r * r))
        J = np.zeros((len(r), 4))
        for k in range(4):
            xp = x.copy()
            xp[k] += 1e-3
            rp2, _, _ = resid(xp)
            J[:, k] = (rp2 - r) / 1e-3
        A = J.T @ J
        g = J.T @ r
        Aaug = A.copy()
        for c in range(4):
            Aaug[c, c] += lam * (A[c, c] + 1e-12)
        try:
            delta = np.linalg.solve(Aaug, g)
        except Exception:
            break
        xtry = x - delta
        r2, _, _ = resid(xtry)
        cost2 = float(np.sum(r2 * r2))
        if cost2 < cost:
            x = xtry
            lam = max(lam / 3, 1e-8)
        else:
            lam = min(lam * 3, 1e4)
    r, F1, F2 = resid(x)
    return complex(x[0], x[1]), complex(x[2], x[3]), F1, F2


def nm_refine(pts, r30, r45, a, t1, t2, ref1, ref2, max_evals=100):
    P = np.array(pts, float)
    lb = np.array([-0.5, 0.5, -0.5, 0.5])
    ub = np.array([0.5, 2.0, 0.5, 2.0])

    def obj(x):
        tt1 = complex(x[0], x[1])
        tt2 = complex(x[2], x[3])
        ptc, pc1, pc2, sc = dd.fast_probs(r30, r45, tt1, tt2, a)
        Fc1, Fc2 = m.extract_foci(pts, pc1, sc[1], pc2, sc[2])
        fc1, fc2 = fit_fast(P, ptc, Fc1, Fc2, a)
        dd_ = fc1 - fc2
        if np.dot(dd_, dd_) < 1e-12:
            return 0.
        dd_ = dd_ / np.linalg.norm(dd_)
        return abs(angle(dd_, ref1) - angle(dd_, ref2))

    x0 = np.clip([t1.real, t1.imag, t2.real, t2.imag], lb, ub)

    def cb(xk):
        return obj(xk) <= 0.5

    try:
        rr = minimize(obj, x0, method="Nelder-Mead", bounds=list(zip(lb, ub)),
                      callback=cb, options={"maxiter": max_evals, "maxfev": max_evals * 2,
                                            "xatol": 1e-6, "fatol": 1e-6})
        best = np.clip(rr.x, lb, ub)
    except Exception:
        best = x0
    obj(best)
    return complex(best[0], best[1]), complex(best[2], best[3])


def run_strategy(name, pts, mode, seed=0, n_rounds=40):
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
    v3 = m.pca(pts)[1]
    v3 = v3 / np.linalg.norm(v3)
    C = P.T @ P
    evals, evecs = np.linalg.eigh(C)
    v_true = evecs[:, np.argsort(evals)[::-1][0]]

    def norm(vv):
        return np.asarray(vv, float) / np.linalg.norm(vv)

    v2 = norm(f1 - f2)
    v4 = norm(f01 - f02)
    F01d = norm(F01 - F02)
    rng = np.random.default_rng(seed)
    t1, t2 = t_, t_
    f1z, f2z = f1, f2
    a9_best = 999.0
    a9t_best = 999.0
    for i in range(n_rounds):
        d_cur = norm(f1z - f2z)
        # 喂给 LM 的 axis 策略
        if mode == "A":
            axis_v = d_cur
        else:
            axis_v = make_axis_at_angle(d_cur, F01d, 60.0, mode, rng)
        t1n, t2n, F1n, F2n = refine_lm(pts, r30, r45, a, t_, t_, axis_v)
        ptn, p1n, p2n, sgn = dd.fast_probs(r30, r45, t1n, t2n, a)
        f1n, f2n = fit_fast(P, ptn, F1n, F2n, a)
        v6 = norm(f1n - f2n)
        a9 = angle(v6, v3)
        a9t = angle(v6, v_true)
        a9_best = min(a9_best, a9)
        a9t_best = min(a9t_best, a9t)
        ang5 = angle(v6, v2)
        ang6 = angle(v6, v4)
        axis = np.cross(v6, v2)
        axis1 = np.cross(v6, v4)
        newf = rotate(v2, axis, -ang5)
        newv = rotate(v4, axis1, -ang6)
        t1, t2 = nm_refine(pts, r30, r45, a, t1n, t2n, newf, newv)
        ptn, p1n, p2n, sgn = dd.fast_probs(r30, r45, t1, t2, a)
        F1n, F2n = m.extract_foci(pts, p1n, sgn[1], p2n, sgn[2])
        f1z, f2z = fit_fast(P, ptn, F1n, F2n, a)
    return a9_best, a9t_best


def main():
    import json
    dats = {}
    with open(r"C:\Users\23128\My project (2)\Assets\Resources\pyjson.json", encoding="utf-8") as f:
        raw = json.load(f)
    dats["pyjson"] = [np.array([p["x"], p["y"], p["z"]], float) for p in raw]
    dats["ball2"] = dd.make_ball(2)
    dats["ellip0"] = dd.make_ellip(0)
    print("=" * 100)
    print("预旋转到60°策略: 变体 A=自指 | B=与参考成60°(同平面) | C=60°锥面随机方位")
    print("=" * 100)
    for name, pts in dats.items():
        print(f"  [{name}]")
        for mode, desc in [("A", "A: axis=d 自指"), ("B", "B: axis 成60°同平面"),
                           ("C", "C: 60°随机方位")]:
            try:
                for seed in ([0] if mode != "C" else [0, 1, 2]):
                    ab, at = run_strategy(name, pts, mode, seed)
                    tag = desc if mode != "C" or seed == 0 else f"C seed{seed}"
                    print(f"    {tag:24s}: a9(v3)best={ab:6.1f}°  a9(真)best={at:6.1f}°",
                          flush=True)
            except Exception as e:
                print(f"    {desc}: 失败 {e}", flush=True)
        print(flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
