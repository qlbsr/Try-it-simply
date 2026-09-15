# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
统计: 每次外循环调用 RefineModuliByAxis 时, 内循环第一次迭代的角度
      angle_first = ∠(F1-F2内部, axis)  (对应 C# angleDegodr)
  问题: 跨外循环, angle_first 是稳步下降 / 上升 / 振荡?
  以及: 内层每次迭代的角度序列 (若 max_iter>1) 是单调下降吗?
"""
import json
import math
import numpy as np

import data_driven_axis as dd
import n2sjy2 as n2
import nsjy_algorithms as m

PYJSON = r"C:\Users\23128\My project (2)\Assets\Resources\pyjson.json"


def angle(u, vv):
    return math.degrees(math.acos(np.clip(np.dot(u, vv) / (np.linalg.norm(u) * np.linalg.norm(vv)), -1, 1)))


def fit_fast(P, prob, F1, F2, a, iters=50, lr=0.0001):
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


def refine_lm_track(pts, r30, r45, a, t10, t20, axis_v, n_iter=50):
    """RefineModuliByAxis + 返回内层每次迭代的角度序列"""
    P = np.array(pts, float)
    axis = np.asarray(axis_v, float)
    axis = axis / np.linalg.norm(axis)
    x = np.array([t10.real, t10.imag, t20.real, t20.imag], float)
    ang_hist = []

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
        dv = F1 - F2
        ang = 180.0
        if np.linalg.norm(dv) > 1e-12:
            dv = dv / np.linalg.norm(dv)
            if np.dot(dv, axis) < 0:
                dv = -dv
            ang = angle(dv, axis)
        ang_hist.append(ang)
        if ang < 0.5:
            break
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
    return complex(x[0], x[1]), complex(x[2], x[3]), F1, F2, ang_hist


def run_trend(name, pts, max_outer=40):
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

    def norm(vv):
        return np.asarray(vv, float) / np.linalg.norm(vv)

    v2 = norm(f1 - f2)
    f01fit = norm(f01 - f02)
    t1, t2 = t_, t_
    f1z, f2z = f1, f2
    first_angles = []     # 每次外循环: 内层第一次角度
    inner_seq = []        # 某次内层的完整序列 (看单调性)
    for i in range(max_outer):
        d_cur = norm(f1z - f2z)
        # 内层 axis = d (用户当前代码)
        t1n, t2n, F1n, F2n, ang_hist = refine_lm_track(pts, r30, r45, a, t_, t_, d_cur)
        first_angles.append(ang_hist[0])
        if i in (0, 1, 2):
            inner_seq.append((i, ang_hist))
        # 更新 (简化: 直接取内层结果)
        ptn, p1n, p2n, sgn = dd.fast_probs(r30, r45, t1n, t2n, a)
        f1n, f2n = fit_fast(P, ptn, F1n, F2n, a)
        f1z, f2z = f1n, f2n
        t1, t2 = t1n, t2n
    return first_angles, inner_seq


def main():
    dats = {}
    with open(PYJSON, "r", encoding="utf-8") as f:
        raw = json.load(f)
    dats["pyjson"] = [np.array([p["x"], p["y"], p["z"]], float) for p in raw]
    for seed in range(3):
        dats[f"ball{seed}"] = dd.make_ball(seed)
    dats["ellip0"] = dd.make_ellip(0)
    dats["ellip3"] = dd.make_ellip(3)

    print("=" * 100)
    print("统计: 每次外循环内层LM第一次角度 (angle_first, 对应 C# angleDegodr)")
    print("=" * 100)
    for name, pts in dats.items():
        try:
            fa, iseq = run_trend(name, pts)
            # 趋势分析: 单调下降段 / 上升 / 振荡
            diffs = [fa[k + 1] - fa[k] for k in range(len(fa) - 1)]
            n_down = sum(1 for d in diffs if d < -0.5)
            n_up = sum(1 for d in diffs if d > 0.5)
            n_flat = sum(1 for d in diffs if abs(d) <= 0.5)
            # 头尾
            print(f"  [{name}] first_angle: 头{fa[0]:5.1f}° → 尾{fa[-1]:5.1f}° "
                  f"| 下降步{n_down} 上升步{n_up} 平{n_flat} (共{len(diffs)}步)")
            # 内层序列单调性 (首次调用)
            if iseq:
                for it_i, seq in iseq[:2]:
                    mono = all(seq[k] <= seq[k + 1] + 1e-6 for k in range(len(seq) - 1))
                    desc = "单调不增" if mono else "非单调"
                    s = ",".join(f"{v:.0f}" for v in seq[:8])
                    print(f"    外环{it_i} 内层角度序列[{len(seq)}]: {s}... → {desc}")
        except Exception as e:
            print(f"  {name}: 失败 {e}", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
