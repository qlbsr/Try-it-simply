# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""line81 调用: RefineModuliByAxis(t_, t_, v3, 50) 不同 wDir:wSelf 的收敛性 (points.json)
用户发现 wDir=1:wSelf=2.5 比 1:1 / 20:1 更容易收敛"""
import numpy as np

import reproduce_cs_outer_faithful as cs
from reproduce_cs_outer_faithful import norm, angle, fit_foci_n2, lattice_probs
import reproduce_cs_new_tau as rn
import nsjy_algorithms as m

d = rn.load_any(r"C:\Users\23128\My project (2)\Assets\Resources\points.json")
P, pts, r30, r45, a = d["P"], d["pts"], d["r30"], d["r45"], d["a"]
v3, v_true = d["v3"], d["v_true"]
init_dir = norm(d["f1"] - d["f2"])
t_ = complex(0, 1)
print(f"points.json  ∠(v3,真轴)={angle(v3, v_true):.2f}°  init_dir∠v3={angle(init_dir, v3):.2f}°")

configs = [
    ("wDir=1 : wSelf=1   (原默认)", 1.0, 1.0),
    ("wDir=1 : wSelf=2.5 (用户发现)", 1.0, 2.5),
    ("wDir=20: wSelf=1   (原Python版)", 20.0, 1.0),
    ("wDir=1 : wSelf=5   ", 1.0, 5.0),
]
for label, wd, ws in configs:
    t1n, t2n, F1n, F2n, odr = cs.refine_moduli_cs(
        pts, r30, r45, a, t_, t_, v3, max_iter=50, capture_odr=True,
        verbose=False, w_dir=wd, w_self=ws)
    odr = odr if odr is not None else 0.0
    # 内层后 ExtractF1-F2 与 v3 夹角 (未经 refit)
    dE = norm(F1n - F2n)
    if np.dot(dE, v3) < 0:
        dE = -dE
    angE = angle(dE, v3)
    # refit
    ptn, p1n, p2n, sgn = lattice_probs(r30, r45, t1n, t2n, a)
    F1x, F2x = m.extract_foci(pts, p1n, sgn[1], p2n, sgn[2])
    f1n, f2n = fit_foci_n2(P, ptn, F1x, F2x, a)
    v6 = norm(f1n - f2n)
    a9 = angle(v6, v3)
    a9t = angle(v6, v_true)
    ok = "✓收敛" if a9 < 10 else ""
    print(f"[{label}] odr={odr:.2f} 内层末∠(Extract,v3)={angE:.2f} | "
          f"refit后 a9(对v3)={a9:.3f} a9真={a9t:.2f} {ok}")
    print(f"          t1n={t1n} t2n={t2n}")
