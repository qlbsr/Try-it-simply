# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
结构自身收敛性扫描: 固定 axi 螺旋扫过半球 (θ: 相对pc1 0~90°, φ: 绕pc1方位),
每点从 (0,1) 冷启跑内层 LM (RefineModuliByAxis 原始版 w_dir=20), 然后 refit.

问题:
  1. cost(axi) 是否有谷? 谷位置是否=结构真方向 (pc1 / 真轴 / 数据不动点)?
  2. 内层能否把 F1-F2 拉到任意 axi (末 angle < tol)? 还是只有部分 axi 可实现?
  3. "结构自身在对应角度有一个收敛" 是否成立 — 若成立, 确定性螺旋扫描即可定位,
     无需自指漂移也无需随机 perturb.
"""
import json
import math
import sys

import numpy as np

import n2sjy2 as n2
import data_driven_axis as dd
import nsjy_algorithms as m
from sklearn.decomposition import PCA

from diag_crawl_mechanism import load, fit_fast_n2, one_round, norm, angle

PYJSON = r"C:\Users\23128\My project (2)\Assets\Resources\pyjson.json"


def basis_around(pc1):
    """构造 ⊥pc1 正交基 e1,e2 (球坐标参考系)"""
    ref = np.array([1.0, 0.0, 0.0]) if abs(pc1[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = norm(np.cross(ref, pc1))
    e2 = norm(np.cross(pc1, e1))
    return e1, e2


def dir_from_polar(pc1, e1, e2, theta_deg, phi_deg):
    th = math.radians(theta_deg)
    ph = math.radians(phi_deg)
    return norm(math.sin(th) * math.cos(ph) * e1
                + math.sin(th) * math.sin(ph) * e2
                + math.cos(th) * pc1)


def scan(dataset="pyjson", thetas=(5, 15, 30, 45, 60, 75, 85, 90),
         phis=(0, 45, 90, 135, 180, 225, 270, 315)):
    d = load(dataset)
    pts, P, r30, r45, a = d["pts"], d["P"], d["r30"], d["r45"], d["a"]
    v_pca = d["v_pca"]
    C = P.T @ P
    evals, evecs = np.linalg.eigh(C)
    v_true = norm(evecs[:, np.argsort(evals)[::-1][0]])
    e1, e2 = basis_around(v_pca)

    rows = []
    t1 = t2 = complex(0, 1)   # 每点冷启 (0,1) — 用一次 (0,1) 概率? 不: refit 概率需该轮的
    print(f"结构扫描 [{dataset}] θ(相对pc1)∈{thetas}°, φ∈{phis}°")
    print(f"  pc1={v_pca} 真轴={v_true}  ∠(pc1,真轴)={angle(v_pca, v_true):.2f}°")
    print(f"  {'θ':>3s} {'φ':>4s} {'axi∠pc1':>8s} {'内层末ang':>9s} {'cost':>10s} "
          f"{'refit∠axi':>9s} {'refit∠pc1':>10s} {'refit∠真':>9s}")
    best = None
    for th in thetas:
        for ph in phis:
            axi = dir_from_polar(v_pca, e1, e2, th, ph)
            # 内层: 从 (0,1) 冷启, 固定 axi
            pt0, _, _, _ = dd.fast_probs(r30, r45, complex(0, 1), complex(0, 1), a)
            # 用原始版 evaluate (快): 概率项用 ExtractF1/F2, w_dir=20
            from run_original_n2sjy2 import refine_silent
            t1n, t2n, F1n, F2n, fa, ea = refine_silent(
                pts, r30, r45, a, complex(0, 1), complex(0, 1), axi, max_iter=50)
            # refit 一步 (原始版: probs(旧 t) → fit)
            pt_cur, _, _, _ = dd.fast_probs(r30, r45, t1n, t2n, a)
            f1n, f2n = fit_fast_n2(P, pt_cur, F1n, F2n, a)
            dir_out = norm(f1n - f2n)
            # 残差 cost 用内层末 cost 重算
            import run_original_n2sjy2 as ro
            r_end, _, _ = n2.evaluate_residuals(
                np.array([t1n.real, t1n.imag, t2n.real, t2n.imag]), pts, axi,
                r30, r45, a, 20.0, 1.0, 1e-3, 1.0, complex(0, 1), complex(0, 1))
            cost_end = float(np.sum(r_end * r_end))
            a_axi = angle(dir_out, axi)
            a_pc = angle(dir_out, v_pca)
            a_tr = angle(dir_out, v_true)
            row = (th, ph, angle(axi, v_pca), ea, cost_end, a_axi, a_pc, a_tr)
            rows.append(row)
            if best is None or cost_end < best[4]:
                best = row
            print(f"  {th:3d} {ph:4d} {row[2]:8.1f} {row[3]:9.2f} {row[4]:10.2f} "
                  f"{row[5]:9.2f} {row[6]:10.2f} {row[7]:9.2f}", flush=True)
    print(f"\n  ==> cost 最小点: θ={best[0]} φ={best[1]} (axi 与pc1夹角={best[2]:.1f}°) "
          f"cost={best[4]:.2f} 内层末ang={best[3]:.2f}° refit方向与pc1={best[6]:.1f}° 与真轴={best[7]:.1f}°")
    return rows, best


if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else "pyjson"
    scan(dataset)
