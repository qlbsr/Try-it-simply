# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
问题: crawl 单调趋势 (angle_pca 每轮 +/-1~2°) 是结构性的还是随机探索?
  A. 每轮 refit 漂移 δ=∠(dir_new, axi) 的旋转轴方向是否恒定 → 结构性
     (若 dir 恒在 axi→pc1 大圆平面内向固定一侧漂移 → 确定性单调, 非随机)
  B. 控制实验: axi' = norm(dir_k + α·pc1切向)  α∈{0(原始), +0.1(拉向pc1), -0.1(推离)}
     → 看 angle_pca 趋势加减是否可控
"""
import json
import math
import sys

import numpy as np

import n2sjy2 as n2
import data_driven_axis as dd
import nsjy_algorithms as m
from sklearn.decomposition import PCA

from run_original_n2sjy2 import refine_silent


def norm(v):
    return np.asarray(v, float) / np.linalg.norm(v)


def angle(u, vv):
    return math.degrees(math.acos(np.clip(np.dot(u, vv) / (np.linalg.norm(u) * np.linalg.norm(vv)), -1, 1)))


def fit_fast_n2(P, prob, F1, F2, a, iters=300, lr=0.0001, lambda_sep=1.0, eps=1e-3):
    F1 = np.asarray(F1, float).copy()
    F2 = np.asarray(F2, float).copy()
    for _ in range(iters):
        d1 = np.linalg.norm(P - F1, axis=1)
        d2 = np.linalg.norm(P - F2, axis=1)
        sd1 = np.maximum(d1, eps)
        sd2 = np.maximum(d2, eps)
        delta = d1 + d2 - 2.0 * a
        fp = np.exp(-np.abs(delta) / (2.0 * a))
        diff = fp - prob
        loss = float(np.sum(diff * diff)) + lambda_sep * float(np.dot(F1 - F2, F1 - F2))
        if loss < 1e-12:
            break
        sign = np.where(delta >= 0, 1.0, -1.0)
        coef = diff * fp * sign / (2.0 * a)
        g1 = coef[:, None] * (P - F1) / sd1[:, None]
        g2 = coef[:, None] * (P - F2) / sd2[:, None]
        n1 = np.linalg.norm(g1, axis=1)
        n2 = np.linalg.norm(g2, axis=1)
        g1 = g1 * np.where(n1 > 1.0, 1.0 / n1, 1.0)[:, None]
        g2 = g2 * np.where(n2 > 1.0, 1.0 / n2, 1.0)[:, None]
        grad1 = g1.sum(0)
        grad2 = g2.sum(0)
        sep = F1 - F2
        grad1 += 2.0 * lambda_sep * sep
        grad2 -= 2.0 * lambda_sep * sep
        gn1 = np.linalg.norm(grad1)
        gn2 = np.linalg.norm(grad2)
        if gn1 > 5.0:
            grad1 *= 5.0 / gn1
        if gn2 > 5.0:
            grad2 *= 5.0 / gn2
        F1 -= lr * grad1
        F2 -= lr * grad2
        if not (np.all(np.isfinite(F1)) and np.all(np.isfinite(F2))):
            break
    return F1, F2


_cache = {}


def load(dataset="pyjson"):
    if dataset in _cache:
        return _cache[dataset]
    if dataset == "pyjson":
        with open(r"C:\Users\23128\My project (2)\Assets\Resources\pyjson.json", "r", encoding="utf-8") as f:
            raw = json.load(f)
        pts = [np.array([p["x"], p["y"], p["z"]], float) for p in raw]
    else:
        np.random.seed(12)
        pts = list(np.random.uniform(-1, 1, (200, 3)))
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
    _cache[dataset] = dict(pts=pts, P=P, r30=r30, r45=r45, a=a,
                           f1=f1, f2=f2, v_pca=v_pca)
    return _cache[dataset]


def one_round(d, t1, t2, axi):
    """原始版一轮: probs(旧t) → refine((0,1)冷启, axi) → fit → dir_new
    返回 (t1n, t2n, dir_new, f1n, f2n)"""
    pts, P, r30, r45, a = d["pts"], d["P"], d["r30"], d["r45"], d["a"]
    pt_cur, _, _, _ = dd.fast_probs(r30, r45, t1, t2, a)
    t1n, t2n, F1n, F2n, _, _ = refine_silent(
        pts, r30, r45, a, complex(0, 1), complex(0, 1), axi, max_iter=50)
    f1n, f2n = fit_fast_n2(P, pt_cur, F1n, F2n, a)
    return t1n, t2n, norm(f1n - f2n)


def run_diag(dataset="pyjson", n_rounds=12):
    d = load(dataset)
    v_pca = d["v_pca"]
    cur_dir = norm(d["f1"] - d["f2"])       # 初始方向
    t1 = t2 = complex(0, 1)
    print(f"\n[A] crawl 方向性诊断 [{dataset}] ∠(init_dir,pc1)={angle(cur_dir, v_pca):.2f}°")
    print(f"  {'it':>3s} {'a_pca':>7s} {'Δa':>6s} {'δ漂移':>7s} {'漂移轴⊥pc1面':>13s}")
    signs = []
    for i in range(n_rounds):
        axi = cur_dir
        a_cur = angle(axi, v_pca)
        t1n, t2n, dnew = one_round(d, t1, t2, axi)
        delta = angle(dnew, axi)
        n_d = np.cross(axi, dnew)
        n_p = np.cross(axi, v_pca)
        ndn = np.linalg.norm(n_d)
        npn = np.linalg.norm(n_p)
        plane = math.nan
        if ndn > 1e-9 and npn > 1e-9:
            plane = math.degrees(math.acos(np.clip(np.dot(n_d, n_p) / (ndn * npn), -1, 1)))
        a_new = angle(dnew, v_pca)
        tang = v_pca - np.dot(v_pca, axi) * axi
        tn = np.linalg.norm(tang)
        sgn = 0.0
        if tn > 1e-9:
            sgn = np.dot(dnew - axi, tang) / tn   # >0 朝 pc1 侧, <0 远离
        signs.append(1 if sgn > 0 else -1)
        print(f"  {i:3d} {a_cur:7.2f} {a_new-a_cur:+6.2f} {delta:7.3f} {plane:13.1f}  sgn={sgn:+.3f}", flush=True)
        cur_dir = dnew
        t1, t2 = t1n, t2n
    print(f"  切向符号序列: {signs}  (全同号=确定性单调; 交替=随机探索)")


def run_control(dataset="pyjson", alpha=0.0, n_rounds=18):
    d = load(dataset)
    v_pca = d["v_pca"]
    cur_dir = norm(d["f1"] - d["f2"])
    t1 = t2 = complex(0, 1)
    tag = "原始α=0" if alpha == 0 else (f"拉向pc1 α=+{alpha}" if alpha > 0 else f"推离pc1 α={alpha}")
    print(f"\n[B] 控制实验 [{dataset}] axi'=norm(dir+α·pc1切向)  {tag}")
    print(f"  {'it':>3s} {'a_pca':>8s} {'Δa':>7s}")
    for i in range(n_rounds):
        axi_raw = cur_dir
        a_cur = angle(axi_raw, v_pca)
        if alpha != 0.0:
            tgt = v_pca - np.dot(v_pca, axi_raw) * axi_raw
            tn = np.linalg.norm(tgt)
            axi = norm(axi_raw + alpha * tgt / tn) if tn > 1e-9 else axi_raw
        else:
            axi = axi_raw
        t1n, t2n, dnew = one_round(d, t1, t2, axi)
        a_new = angle(dnew, v_pca)
        if i % 2 == 0:
            print(f"  {i:3d} {a_new:8.2f} {a_new-a_cur:+7.2f}", flush=True)
        cur_dir = dnew
        t1, t2 = t1n, t2n
    return angle(cur_dir, v_pca)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "diag"):
        run_diag("pyjson", 14)
    if which in ("all", "ctl"):
        for al in [0.0, 0.1, -0.1]:
            run_control("pyjson", alpha=al, n_rounds=18)
