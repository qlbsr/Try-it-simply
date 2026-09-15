# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
对照 C# pca() 的 v3 重建 vs Python m.pca 的 v3 重建
C# (用户贴的):
  pc1 = 第一主成分 (PCAScikitLearn Components[0,:] = (x,y,z))
  fj  = atan2(pc1.z, pc1.x)   ← z/x!
  d2  = acos(pc1.y)
  v3  = (sin d2 cos fj, cos d2, sin d2 sin fj)
Python m.pca (nsjy_algorithms.py 308-312):
  fj = atan2(pc1[1], pc1[0])  ← y/x  (错位!)
  d2 = acos(pc1[1])
"""
import json
import math

import numpy as np

import n2sjy2 as n2
import nsjy_algorithms as m


def norm(v):
    return np.asarray(v, float) / np.linalg.norm(v)


def angle(u, vv):
    return math.degrees(math.acos(np.clip(np.dot(u, vv) / (np.linalg.norm(u) * np.linalg.norm(vv)), -1, 1)))


def pca_cs(points):
    """逐行对应 C# pca(): pc1 用与 m.pca 相同的 SVD/eigh 主轴, 但 v3 重建用 atan2(z,x)"""
    X = np.asarray(points, float)
    Xc = X - X.mean(axis=0)
    cov = Xc.T @ Xc / (len(X) - 1)
    evals, evecs = np.linalg.eigh(cov)
    pc1 = evecs[:, np.argsort(evals)[::-1][0]]
    fj = math.atan2(pc1[2], pc1[0])          # C#: atan2(pc1.z, pc1.x)
    d2 = math.acos(np.clip(pc1[1], -1, 1))   # C#: acos(pc1.y)
    v3 = np.array([math.sin(d2) * math.cos(fj),
                   math.cos(d2),
                   math.sin(d2) * math.sin(fj)])
    return norm(pc1), norm(v3)


def main():
    with open(r"C:\Users\23128\My project (2)\Assets\Resources\pyjson.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    pts = [np.array([p["x"], p["y"], p["z"]], float) for p in raw]
    P = np.array(pts, float)

    pc1_py, v3_py = m.pca(pts)[0], norm(m.pca(pts)[1])
    pc1_cs, v3_cs = pca_cs(pts)
    C = P.T @ P
    evals, evecs = np.linalg.eigh(C)
    v_true = norm(evecs[:, np.argsort(evals)[::-1][0]])

    print(f"pc1 (Python主轴) = {pc1_py}")
    print(f"pc1 (C# 主轴同源) = {pc1_cs}  差={np.linalg.norm(pc1_py-pc1_cs):.2e}")
    print()
    print(f"v3 Python (atan2(y,x)) = {v3_py}")
    print(f"v3 C#     (atan2(z,x)) = {v3_cs}")
    print(f"∠(v3_py, v3_cs) = {angle(v3_py, v3_cs):.2f}°")
    print()
    print(f"真主轴 (PᵀP max eig)  = {v_true}")
    print(f"∠(v3_py, 真轴) = {angle(v3_py, v_true):.2f}°")
    print(f"∠(v3_cs, 真轴) = {angle(v3_cs, v_true):.2f}°")
    print(f"∠(pc1, 真轴)   = {angle(pc1_py, v_true):.2f}°  (未中心化 C=PᵀP vs 中心化 cov)")


if __name__ == "__main__":
    main()
