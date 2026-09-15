# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
验证: C# pca() 的 v3 (球坐标重建) 与 sklearn pc1 (pca_axis) 是否同一方向?
C#: pc1 = PCAScikitLearn.Components[0] (单位向量)
     fj = atan2(pc1.z, pc1.x);  d2 = acos(pc1.y)
     v3 = (sin d2 cos fj, cos d2, sin d2 sin fj)
数学: 单位向量 (x,y,z): sin d2 = sqrt(1-y^2)=|sqrt(x^2+z^2)|,
      cos fj = x/sqrt(x^2+z^2), sin fj = z/sqrt(x^2+z^2)
      → v3 = (x, y, z) = pc1 恒等 (当 |y|<=1, 且 sqrt 取正时 cos/sin 保留符号)
"""
import json
import math

import numpy as np

import n2sjy2 as n2
import nsjy_algorithms as m

PYJSON = r"C:\Users\23128\My project (2)\Assets\Resources\pyjson.json"


def norm(v):
    return np.asarray(v, float) / np.linalg.norm(v)


def angle(u, vv):
    return math.degrees(math.acos(np.clip(np.dot(u, vv) / (np.linalg.norm(u) * np.linalg.norm(vv)), -1, 1)))


def v3_from_pc1(pc1):
    """C# pca() 的重建公式, 输入任意 pc1"""
    fj = math.atan2(pc1[2], pc1[0])
    d2 = math.acos(np.clip(pc1[1], -1, 1))
    return np.array([math.sin(d2) * math.cos(fj),
                     math.cos(d2),
                     math.sin(d2) * math.sin(fj)])


def pca_sklearn(points):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    pca.fit(np.array(points))
    return norm(pca.components_[0])


def datasets():
    with open(PYJSON, "r", encoding="utf-8") as f:
        raw = json.load(f)
    yield "pyjson", [np.array([p["x"], p["y"], p["z"]], float) for p in raw]
    rng = np.random.default_rng(12)
    yield "seed12-uniform", list(rng.uniform(-1, 1, (200, 3)))


for name, pts in datasets():
    P = np.array(pts, float)
    # sklearn pc1 (原版 n2sjy2.pca_axis)
    sk = pca_sklearn(pts)
    # 中心化 eigh 主轴 (m.pca 修复版内部 pc1)
    pc1_m = m.pca(pts)[0]
    pc1_m = norm(pc1_m)
    # C# 重建公式作用于 sklearn pc1
    v3_from_sk = norm(v3_from_pc1(sk))
    # C# 重建作用于 eigh pc1 (m.pca 修复后的 v3)
    v3_m = norm(m.pca(pts)[1])
    # 未中心化 C=P^T P 主轴
    evals, evecs = np.linalg.eigh(P.T @ P)
    v_true = norm(evecs[:, np.argsort(evals)[::-1][0]])
    # 中心化协方差主轴
    Xc = P - P.mean(axis=0)
    evc, evc_ = np.linalg.eigh(Xc.T @ Xc / (len(P) - 1))
    v_center = norm(evc_[:, np.argsort(evc)[::-1][0]])

    print(f"\n===== {name} =====")
    print(f"sklearn pc1        = {sk}")
    print(f"m.pca pc1 (eigh)   = {pc1_m}")
    print(f"∠(sklearn, eigh)   = {angle(sk, pc1_m):.4f}°   (符号可能差180)")
    print(f"∠(sklearn, -eigh)  = {angle(sk, -pc1_m):.4f}°")
    print(f"v3(重建自sklearn)  = {v3_from_sk}")
    print(f"∠(v3_from_sk, sk)  = {angle(v3_from_sk, sk):.2e}°   ← 重建≡pc1?")
    print(f"v3(m.pca修复后)    = {v3_m}")
    print(f"∠(v3_m, pc1_m)     = {angle(v3_m, pc1_m):.2e}°")
    print(f"--- 参照轴对比 ---")
    print(f"∠(sk, 中心化主轴)  = {angle(sk, v_center):.4f}°")
    print(f"∠(sk, 未中心C主轴) = {angle(sk, v_true):.4f}°  (点云均值偏移造成的差)")
    print(f"∠(v3_m, v_true)    = {angle(v3_m, v_true):.4f}°")
