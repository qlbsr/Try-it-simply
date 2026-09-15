# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""直接对 pcav(v3) 收敛测试: RefineModuliByAxis(axis=v3) → refit → a9?"""
import numpy as np

import reproduce_cs_outer_faithful as cs
from reproduce_cs_outer_faithful import norm, angle, fit_foci_n2, lattice_probs
import reproduce_cs_new_tau as rn
import nsjy_algorithms as m

d = rn.load_any(r"C:\Users\23128\My project (2)\Assets\Resources\points.json")
P, pts, r30, r45, a = d["P"], d["pts"], d["r30"], d["r45"], d["a"]
v3, v_true = d["v3"], d["v_true"]
init_dir = norm(d["f1"] - d["f2"])
print(f"v3={v3}  ∠(v3,真轴)={angle(v3, v_true):.2f}°  init_dir∠v3={angle(init_dir, v3):.2f}°")

starts = [
    ("(0,1)初值", complex(0, 1), complex(0, 1)),
    ("理论tau初值", *rn.n2.compute_taus()),
    ("NM收敛值(0.323,0.985)", complex(0.323141198796178, 0.984532062400871),
     complex(0.292181566064164, 1.00644754227822)),
]
for label, t10, t20 in starts:
    t1n, t2n, F1n, F2n, odr = cs.refine_moduli_cs(
        pts, r30, r45, a, t10, t20, v3, max_iter=50, capture_odr=True, verbose=False)
    odr = odr if odr is not None else 0.0
    ptn, p1n, p2n, sgn = lattice_probs(r30, r45, t1n, t2n, a)
    F1x, F2x = m.extract_foci(pts, p1n, sgn[1], p2n, sgn[2])
    f1n, f2n = fit_foci_n2(P, ptn, F1x, F2x, a)
    v6 = norm(f1n - f2n)
    a9 = angle(v6, v3)
    a5 = angle(v6, init_dir)
    print(f"[{label}] axis=v3: t1n={t1n} t2n={t2n} odr={odr:.2f} "
          f"refit后 a9(对v3)={a9:.3f} a9真={angle(v6, v_true):.2f} a5={a5:.2f}")
