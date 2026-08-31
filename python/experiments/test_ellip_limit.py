# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""ellip0 内层极限测试: {150, 200} 冷启动, 外环25, 看 |Δ| 与 angle_pca 的极限"""
import math

import numpy as np

import data_driven_axis as dd
import n2sjy2 as n2
import test_inner_cold as tic

pts = dd.make_ellip(0)
d0 = dd.init_direction(pts)
print("ellip0 内层极限 (冷启动, 外环25):", flush=True)
for inner in (150, 200):
    rec = tic.outer_cold(pts, d0, inner, outer=25)
    a1, ap = rec[-1]
    a1_0, ap_0 = rec[0]
    dmin = min(abs(r[0] - r[1]) for r in rec)
    print(f"  内层={inner:3d}: 首(angle1={a1_0:5.1f}, angle_pca={ap_0:5.1f}) → "
          f"末(angle1={a1:5.1f}, angle_pca={ap:5.1f})  |Δ|end={abs(a1-ap):5.1f}°  "
          f"全程min|Δ|={dmin:5.1f}°", flush=True)
print("DONE", flush=True)
