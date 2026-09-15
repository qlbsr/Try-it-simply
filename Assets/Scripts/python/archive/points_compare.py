# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
points.json 对比: 原始 n2sjy2py (45° 停滞?) vs C# n2sjy2 忠实 (168° 停, 距180仅12?)
"""
import json
import math
import sys
import time

import numpy as np

import n2sjy2 as n2
import data_driven_axis as dd
import nsjy_algorithms as m
from sklearn.decomposition import PCA

from run_original_n2sjy2 import refine_silent, norm, angle
from diag_crawl_mechanism import fit_fast_n2
import reproduce_cs_outer_faithful as cs

POINTS = r"C:\Users\23128\My project (2)\Assets\Resources\points.json"


def load_points(path=POINTS):
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
    pt0, p10, p20, sg0 = dd.fast_probs(r30, r45, t1t, t2t, a)
    F1i, F2i = m.extract_foci(pts, p10, sg0[1], p20, sg0[2])
    f1, f2 = fit_fast_n2(P, pt0, F1i, F2i, a)
    pca = PCA(n_components=3)
    pca.fit(P)
    v_pca = norm(pca.components_[0])
    C = P.T @ P
    evals, evecs = np.linalg.eigh(C)
    v_true = norm(evecs[:, np.argsort(evals)[::-1][0]])
    return dict(pts=pts, P=P, r30=r30, r45=r45, a=a, f1=f1, f2=f2,
                v_pca=v_pca, v_true=v_true, init_dir=norm(f1 - f2))


# ---------- 原始 Python 版主循环 ----------
def run_py_original(d, max_outer=80):
    v_pca = d["v_pca"]
    cur = norm(d["f1"] - d["f2"])
    t1 = t2 = complex(0, 1)
    print(f"\n[原始Python版] points.json  ∠(init_dir,pc1)={angle(cur, v_pca):.2f}°  "
          f"∠(pc1,真轴)={angle(v_pca, d['v_true']):.2f}°")
    print(f"  {'it':>3s} {'angle1':>7s} {'angle_pca':>9s} {'|Δ|':>6s} {'a9v3':>7s} {'首内ang':>8s}")
    stop = None
    for i in range(max_outer):
        axi = cur
        t1n, t2n, dnew = None, None, None
        pt_cur, _, _, _ = dd.fast_probs(r30 := d["r30"], r45 := d["r45"], t1, t2, d["a"])
        t1n, t2n, F1n, F2n, fa, ea = refine_silent(
            d["pts"], r30, r45, d["a"], complex(0, 1), complex(0, 1), axi, max_iter=50)
        f1n, f2n = fit_fast_n2(d["P"], pt_cur, F1n, F2n, d["a"])
        dnew = norm(f1n - f2n)
        a1 = angle(dnew, d["init_dir"])
        ap = angle(dnew, v_pca)
        a9 = angle(dnew, d["v_true"])
        flag = abs(a1 - ap) <= 16
        if i % 2 == 0 or flag:
            print(f"  {i:3d} {a1:7.2f} {ap:9.2f} {abs(a1-ap):6.2f} {a9:7.2f} {fa if fa is not None else 0:8.1f}", flush=True)
        cur = dnew
        t1, t2 = t1n, t2n
        if flag:
            stop = i
            break
    print(f"  ==> py原版 stop@{stop} | 末轮 angle1={a1:.2f} angle_pca={ap:.2f} (对真轴 {a9:.2f})")
    return ap


# ---------- C# 忠实完整外环 ----------
def run_cs(d, max_outer=30, use_cond2=False):
    P = d["P"]
    pts = d["pts"]
    r30, r45, a = d["r30"], d["r45"], d["a"]
    t_ = complex(0, 1)
    v3 = norm(m.pca(pts)[1])
    v_true = d["v_true"]
    pt01, p101, p201, sig01 = cs.lattice_probs(r30, r45, t_, t_, a)
    F01, F02 = m.extract_foci(pts, p101, sig01[1], p201, sig01[2])
    f01, f02 = cs.fit_foci_n2(P, pt01, F01, F02, a)
    v4 = norm(f01 - f02)
    f1, f2 = d["f1"], d["f2"]
    v2 = norm(f1 - f2)
    print(f"\n[C#忠实版] points.json  ∠(v3,真轴)={angle(v3, v_true):.2f}°  v4(f01-f02)∠pc1={angle(v4, d['v_pca']):.2f}°")
    print(f"  {'it':>3s} {'a9(v3)':>8s} {'a9真':>8s} {'odr':>7s} {'new':>7s} {'|Δ|':>7s} {'stop?':>5s}")
    stop = None
    for i in range(max_outer):
        d_cur = norm(f1 - f2)
        ang_new = angle(d_cur, v4)
        axi = d_cur
        if ang_new < 5:
            n_ax = np.cross(d_cur, v4)
            if np.linalg.norm(n_ax) < 1e-12:
                n_ax = np.cross(d_cur, np.array([0., 1., 0.]))
            axi = norm(cs.rotate_deg(d_cur, n_ax, 60.0))
        t1n, t2n, F1n, F2n, odr = cs.refine_moduli_cs(
            pts, r30, r45, a, t_, t_, axi, max_iter=50, capture_odr=True, verbose=False)
        odr = odr if odr is not None else 0.0
        ptn, p1n, p2n, sgn = cs.lattice_probs(r30, r45, t1n, t2n, a)
        F1x, F2x = m.extract_foci(pts, p1n, sgn[1], p2n, sgn[2])
        f1n, f2n = cs.fit_foci_n2(P, ptn, F1x, F2x, a)
        v6 = norm(f1n - f2n)
        a9v = angle(v6, v3)
        a9t = angle(v6, v_true)
        a5 = angle(v6, v2)
        a6 = angle(v6, v4)
        diff = abs(odr - ang_new)
        cond = diff > 16
        stop_flag = (cond and a9v < 1) if use_cond2 else cond
        if i % 2 == 0 or stop_flag:
            print(f"  {i:3d} {a9v:8.2f} {a9t:8.2f} {odr:7.2f} {ang_new:7.2f} {diff:7.2f} {str(stop_flag):>5s}  a5={a5:6.2f} |a9-a5|={abs(a9v-a5):6.2f}", flush=True)
        if stop_flag:
            stop = i
            break
        ang5 = angle(v6, v2)
        ang6 = angle(v6, v4)
        axis = np.cross(v6, v4)
        axis1 = np.cross(v6, v2)
        newf = norm(cs.rotate_deg(v4, axis, -ang5))
        newv = norm(cs.rotate_deg(v2, axis1, -ang6))
        t1_, t2_, delta, F1z, F2z = cs.nm_refine_cs(pts, r30, r45, a, t1n, t2n, newf, newv)
        f1, f2 = F1z, F2z
    print(f"  ==> C#版 stop@{stop} | 末轮 a9(v3)={a9v:.2f} 真={a9t:.2f} odr={odr:.2f} new={ang_new:.2f}")
    return a9v, a9t


if __name__ == "__main__":
    d = load_points()
    print(f"points.json n={len(d['pts'])} a={d['a']:.4f} rp 已含")
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("both", "py"):
        run_py_original(d, max_outer=80)
    if which in ("both", "cs"):
        run_cs(d, max_outer=30, use_cond2=False)
