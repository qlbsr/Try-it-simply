# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
复现用户修改后的 C# n2sjy2 (新版):
  1. t1,t2 初始 (0,1), 每轮 NM 后更新 → RefineModuliByAxis 热启动 (不再每轮 (0,1) 冷启)
  2. t1/t2 同时作为 RefineModuliByAxis 理论正则锚点 (EvaluateResiduals r[n+3..n+6]) — 锚点随迭代漂移
  3. 终止: a9 = ∠(v6, v3) < 10 → break  (不再是 |odr-new|>16)
  4. 数据集: points1.json (Unity line 33 "points1")
对照旧版 ((0,1) 冷启) 在 points.json 上停 45°(py)/168°(cs) 的行为
"""
import json
import math
import sys

import numpy as np

import n2sjy2 as n2
import data_driven_axis as dd
import nsjy_algorithms as m
from sklearn.decomposition import PCA

import reproduce_cs_outer_faithful as cs
from reproduce_cs_outer_faithful import (norm, angle, rotate_deg, fit_foci_n2,
                                         lattice_probs, normalize_tau_cs,
                                         evaluate_residuals_cs, refine_moduli_cs)
from nelder_mead_cs import nelder_mead_cs


def nm_refine_cs_exact(pts, r30, r45, a, t1i, t2i, ref_newf, ref_newv,
                       max_evals=100, stop_f=16.0):
    """RefineTausWithNM 用 C# NelderMead 精确实现 (n2sjy2.cs 180-233)"""
    P = np.asarray(pts, float)
    lb = np.array([-0.5, 0.5, -0.5, 0.5])
    ub = np.array([0.5, 2.0, 0.5, 2.0])
    state = {"F1z": None, "F2z": None}

    def obj(x):
        tt1 = complex(x[0], x[1])
        tt2 = complex(x[2], x[3])
        ptc, pc1, pc2, sc = lattice_probs(r30, r45, tt1, tt2, a)
        Fc1, Fc2 = m.extract_foci(pts, pc1, sc[1], pc2, sc[2])
        fc1, fc2 = fit_foci_n2(P, ptc, Fc1, Fc2, a)
        state["F1z"], state["F2z"] = fc1, fc2
        d = fc1 - fc2
        if np.dot(d, d) < 1e-12:
            return 0.0
        d = d / np.linalg.norm(d)
        ad = angle(d, ref_newf)
        ad1 = angle(d, ref_newv)
        return abs(ad - ad1)

    x0 = np.clip([t1i.real, t1i.imag, t2i.real, t2i.imag], lb, ub)
    bx, bf, evals = nelder_mead_cs(obj, x0, lb, ub, max_evals=max_evals,
                                   stop_f=stop_f, tol=1e-6)
    val = obj(bx)
    return (complex(bx[0], bx[1]), complex(bx[2], bx[3]),
            val, state["F1z"], state["F2z"])


def load_any(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw[0], dict):
        pts = [np.array([p["x"], p["y"], p["z"]], float) for p in raw]
    else:
        pts = [np.array(p, float) for p in raw]
    P = np.array(pts, float)
    rp = n2.compute_rp(pts)
    d2 = math.asin(math.cos(math.radians(30)) / math.pi)
    a = rp * (1 + math.sin(d2))
    _, r45, r30, _ = n2.yzqx(pts, rp)
    t1t, t2t = n2.compute_taus()
    pt0, p10, p20, sg0 = lattice_probs(r30, r45, t1t, t2t, a)
    F1i, F2i = m.extract_foci(pts, p10, sg0[1], p20, sg0[2])
    f1, f2 = fit_foci_n2(P, pt0, F1i, F2i, a)
    pc1, v3, _ = m.pca(pts)
    C = P.T @ P
    evals, evecs = np.linalg.eigh(C)
    v_true = norm(evecs[:, np.argsort(evals)[::-1][0]])
    return dict(pts=pts, P=P, r30=r30, r45=r45, a=a, f1=f1, f2=f2,
                v3=norm(v3), v_true=v_true)


def run_new(d, max_outer=60, tag="", init_tau="unit"):
    P, pts, r30, r45, a = d["P"], d["pts"], d["r30"], d["r45"], d["a"]
    v3, v_true = d["v3"], d["v_true"]
    t_ = complex(0, 1)
    # --- line 61-71: 初始拟合 (与 C# 相同) ---
    pt01, p101, p201, sig01 = lattice_probs(r30, r45, t_, t_, a)
    F01, F02 = m.extract_foci(pts, p101, sig01[1], p201, sig01[2])
    f01, f02 = fit_foci_n2(P, pt01, F01, F02, a)
    v4 = norm(f01 - f02)
    f1, f2 = d["f1"], d["f2"]
    v2 = norm(f1 - f2)
    # --- line 79-80: t1 = t_ = (0,1); 变体 init_tau="theory" 用理论 taus ---
    if init_tau == "theory":
        t1, t2 = n2.compute_taus()
    else:
        t1 = t2 = complex(0, 1)
    print(f"\n{tag}新版 C# (热启动t+锚点漂移+a9<10停, init={init_tau})  ∠(v3,真轴)={angle(v3, v_true):.2f}°")
    print(f"  {'it':>3s} {'a9(v3)':>8s} {'a5':>7s} {'odr':>7s} {'new':>7s} {'t1':>20s} {'t2':>20s} {'stop?':>5s}")
    stop = None
    for i in range(max_outer):
        axi = norm(f1 - f2)
        ang_new = angle(axi, v4)
        # --- line 125: RefineModuliByAxis(points, t1, t2, axi, 50)  热启动! ---
        t1n, t2n, F1n, F2n, odr = refine_moduli_cs(
            pts, r30, r45, a, t1, t2, axi, max_iter=50, capture_odr=True, verbose=False)
        odr = odr if odr is not None else 0.0
        # --- line 126-128 ---
        ptn, p1n, p2n, sgn = lattice_probs(r30, r45, t1n, t2n, a)
        F1x, F2x = m.extract_foci(pts, p1n, sgn[1], p2n, sgn[2])
        f1n, f2n = fit_foci_n2(P, ptn, F1x, F2x, a)
        v6 = norm(f1n - f2n)
        a9 = angle(v6, v3)
        a5 = angle(v6, v2)
        flag = a9 < 10
        if i % 2 == 0 or flag:
            print(f"  {i:3d} {a9:8.2f} {a5:7.2f} {odr:7.2f} {ang_new:7.2f} "
                  f"{t1n.real:+.3f}{t1n.imag:+.3f}j {t2n.real:+.3f}{t2n.imag:+.3f}j {str(flag):>5s}", flush=True)
        if flag:
            stop = i
            break
        # --- line 140-148: 旋转 + NM ---
        ang5 = angle(v6, v2)
        ang6 = angle(v6, v4)
        axis = np.cross(v6, v4)
        axis1 = np.cross(v6, v2)
        newf = norm(rotate_deg(v4, axis, -ang5))
        newv = norm(rotate_deg(v2, axis1, -ang6))
        t1_, t2_, delta, F1z, F2z = nm_refine_cs_exact(pts, r30, r45, a, t1n, t2n, newf, newv)
        # --- line 150-155: f1/f2 与 t1/t2 更新 ---
        f1, f2 = F1z, F2z
        t1, t2 = t1_, t2_
    print(f"  ==> stop@{stop} | 末轮 a9={a9:.2f} (真 {angle(v6, v_true):.2f}) t1={t1} t2={t2}")
    return a9


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\23128\My project (2)\Assets\Resources\points1.json"
    maxo = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    d = load_any(path)
    print(f"数据集: {path} n={len(d['pts'])}")
    run_new(d, max_outer=maxo, tag="[NEW] ")
