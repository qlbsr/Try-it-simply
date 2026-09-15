# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
原始 n2sjy2.py 主循环语义跑起来 (静音/加速版), 记录 angle1/angle_pca 轨迹
与 C# n2sjy2 (reproduce_cs_outer_faithful) 对比角度增长趋势

原版语义 (n2sjy2.py 412-470):
  - 每轮: prob_total_new = probs(上一轮 t1,t2)   [旧 t!]
          refine_moduli_by_axis(从(0,1)冷启, axis=f1-f2, 50 iter)  [残差概率项用 ExtractF1/F2 未先fit, w_dir=20]
          f1_new,f2_new = fit(prob_total_new, F1_new, F2_new)
  - dir_vec=(f1_new-f2_new).n; angle1=∠(dir,init_dir); angle_pca=∠(dir,pc1)
  - 终止 |angle1-angle_pca|<=16
"""
import json
import math
import sys
import time

import numpy as np

import n2sjy2 as n2
import data_driven_axis as dd          # patch n2.compute_probabilities_from_taus=fast_probs
import nsjy_algorithms as m
from sklearn.decomposition import PCA


def norm(v):
    return np.asarray(v, float) / np.linalg.norm(v)


def angle(u, vv):
    return math.degrees(math.acos(np.clip(np.dot(u, vv) / (np.linalg.norm(u) * np.linalg.norm(vv)), -1, 1)))


def pca_axis(points):
    data = np.array(points)
    pca = PCA(n_components=3)
    pca.fit(data)
    return norm(pca.components_[0])


# ---------- 静音版 refine (复制 n2sjy2.refine_moduli_by_axis 340-408, 无打印) ----------
def refine_silent(pts, r30, r45, a, t10, t20, pca_axis_v, max_iter=50,
                  angle_tol_deg=0.5, fd_h=1e-3, w_dir=20.0, w_self=1.0,
                  w_theory=1e-3, w_i=1.0):
    axis = np.asarray(pca_axis_v, float)
    axis = axis / np.linalg.norm(axis)
    x = np.array([t10.real, t10.imag, t20.real, t20.imag], float)
    lam = 1e-3
    first_ang = None
    F1 = F2 = np.zeros(3)
    for it in range(max_iter):
        r, F1, F2 = n2.evaluate_residuals(x, pts, axis, r30, r45, a,
                                          w_dir, w_self, w_theory, w_i, t10, t20)
        cost = float(np.sum(r * r))
        dv = F1 - F2
        ang = 180.0
        if np.linalg.norm(dv) > 1e-12:
            dv = dv / np.linalg.norm(dv)
            if np.dot(dv, axis) < 0:
                dv = -dv
            ang = angle(dv, axis)
            if first_ang is None:
                first_ang = ang
        if ang < angle_tol_deg:
            break
        J = np.zeros((len(r), 4))
        for k in range(4):
            xp = x.copy()
            xp[k] += fd_h
            n2.normalize_tau(xp)
            rp2, _, _ = n2.evaluate_residuals(xp, pts, axis, r30, r45, a,
                                              w_dir, w_self, w_theory, w_i, t10, t20)
            J[:, k] = (rp2 - r) / fd_h
        A = J.T @ J
        g = J.T @ r
        A_aug = A.copy()
        for c in range(4):
            A_aug[c, c] += lam * (A[c, c] + 1e-12)
        try:
            delta = np.linalg.solve(A_aug, g)
        except np.linalg.LinAlgError:
            break
        xtry = x - delta
        n2.normalize_tau(xtry)
        r2, _, _ = n2.evaluate_residuals(xtry, pts, axis, r30, r45, a,
                                         w_dir, w_self, w_theory, w_i, t10, t20)
        cost2 = float(np.sum(r2 * r2))
        if cost2 < cost:
            x = xtry
            lam = max(lam / 3.0, 1e-8)
        else:
            lam = min(lam * 3.0, 1e4)
    r, F1, F2 = n2.evaluate_residuals(x, pts, axis, r30, r45, a,
                                      w_dir, w_self, w_theory, w_i, t10, t20)
    dv = F1 - F2
    end_ang = 180.0
    if np.linalg.norm(dv) > 1e-12:
        dv = dv / np.linalg.norm(dv)
        if np.dot(dv, axis) < 0:
            dv = -dv
        end_ang = angle(dv, axis)
    return complex(x[0], x[1]), complex(x[2], x[3]), F1, F2, first_ang, end_ang


def run_original(dataset, max_outer=100, tag=""):
    if dataset == "pyjson":
        with open(r"C:\Users\23128\My project (2)\Assets\Resources\pyjson.json", "r", encoding="utf-8") as f:
            raw = json.load(f)
        pts = [np.array([p["x"], p["y"], p["z"]], float) for p in raw]
    else:  # seed12 uniform (原版默认)
        np.random.seed(12)
        pts = list(np.random.uniform(-1, 1, (200, 3)))

    P = np.array(pts, float)
    rp = n2.compute_rp(pts)
    d2 = math.asin(math.cos(math.radians(30)) / math.pi)
    a = rp * (1 + math.sin(d2))
    _, r45, r30, _ = n2.yzqx(pts, rp)
    t1t, t2t = n2.compute_taus()
    pc1 = pca_axis(pts)
    v3 = norm(m.pca(pts)[1])            # C# 语义 v3 (=pc1 修复后)

    pt0, p10, p20, sg0 = dd.fast_probs(r30, r45, t1t, t2t, a)
    F1i, F2i = m.extract_foci(pts, p10, sg0[1], p20, sg0[2])
    f1, f2 = n2.fit_foci_by_probability(pts, pt0, F1i, F2i, a,
                                        max_iter=300, learning_rate=0.0001, lambda_sep=1.0)
    f1 = np.asarray(f1, float)
    f2 = np.asarray(f2, float)
    init_dir = norm(f1 - f2)
    v_pca = pc1
    t1, t2 = complex(0, 1), complex(0, 1)
    print(f"\n{tag}原始n2sjy2主循环 [{dataset}] n={len(pts)} 终止|angle1-angle_pca|<=16:")
    print(f"  pc1={v_pca}  init_dir={init_dir}  ∠(init_dir,pc1)={angle(init_dir, v_pca):.2f}°")
    print(f"  {'iter':>4s} {'angle1':>8s} {'angle_pca':>10s} {'|Δ|':>7s} {'a9v3':>8s} {'首内ang':>8s} {'末内ang':>8s} {'stop?':>5s}")
    stop = None
    t0 = time.time()
    for i in range(max_outer):
        pt_cur, _, _, _ = dd.fast_probs(r30, r45, t1, t2, a)      # 旧 t 概率
        axis = norm(f1 - f2)
        t1n, t2n, F1n, F2n, fa, ea = refine_silent(
            pts, r30, r45, a, complex(0, 1), complex(0, 1), axis, max_iter=50)
        f1n, f2n = n2.fit_foci_by_probability(pts, pt_cur, F1n, F2n, a,
                                              max_iter=300, learning_rate=0.0001, lambda_sep=1.0)
        f1n = np.asarray(f1n, float)
        f2n = np.asarray(f2n, float)
        dir_vec = norm(f1n - f2n)
        a1 = angle(dir_vec, init_dir)
        ap = angle(dir_vec, v_pca)
        a9 = angle(dir_vec, v3)
        flag = abs(a1 - ap) <= 16
        if i % 2 == 0 or flag:
            print(f"  {i:4d} {a1:8.2f} {ap:10.2f} {abs(a1-ap):7.2f} {a9:8.2f} "
                  f"{fa if fa is not None else 0:8.1f} {ea:8.1f} {str(flag):>5s}", flush=True)
        f1, f2 = f1n, f2n
        t1, t2 = t1n, t2n
        if flag:
            stop = i
            break
    print(f"  ==> stop@iter{stop} ({time.time()-t0:.0f}s) | 末轮 angle1={a1:.2f} angle_pca={ap:.2f} a9={a9:.2f}")
    return stop, a1, ap, a9


if __name__ == "__main__":
    mo = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    run_original("seed12", max_outer=mo, tag="[S] ")
    run_original("pyjson", max_outer=mo, tag="[P] ")
