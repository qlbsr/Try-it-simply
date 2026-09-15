# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
扫描 1:2.5 权重下 line81 (axis=v3, 从(0,1), 50 iter) 的收敛性
收敛判据: refit 后 a9 = ∠(v6, v3) < 10  (Unity line 139)
不收敛的点集保存到 records/bad_ws25/
"""
import json
import math
import os
import time

import numpy as np

import n2sjy2 as n2
import nsjy_algorithms as m
import reproduce_cs_outer_faithful as cs
from reproduce_cs_outer_faithful import norm, angle, fit_foci_n2, lattice_probs

RES = r"C:\Users\23128\My project (2)\Assets\Resources"
OUT = r"C:\Users\23128\My project (2)\Assets\Scripts\python\experiments\records"
os.makedirs(OUT, exist_ok=True)

W_DIR, W_SELF = 1.0, 2.5


def test_dataset(pts, name):
    P = np.array(pts, float)
    rp = n2.compute_rp(pts)
    d2 = math.asin(math.cos(math.radians(30)) / math.pi)
    a = rp * (1 + math.sin(d2))
    _, r45, r30, _ = n2.yzqx(pts, rp)
    pc1, v3, _ = m.pca(pts)
    v3 = norm(v3)
    t_ = complex(0, 1)
    t0 = time.time()
    t1n, t2n, F1n, F2n, odr = cs.refine_moduli_cs(
        pts, r30, r45, a, t_, t_, v3, max_iter=50, capture_odr=True,
        w_dir=W_DIR, w_self=W_SELF)
    odr = odr if odr is not None else 0.0
    ptn, p1n, p2n, sgn = lattice_probs(r30, r45, t1n, t2n, a)
    F1x, F2x = m.extract_foci(pts, p1n, sgn[1], p2n, sgn[2])
    f1n, f2n = fit_foci_n2(P, ptn, F1x, F2x, a)
    v6 = norm(f1n - f2n)
    a9 = angle(v6, v3)
    dE = norm(F1n - F2n)
    if np.dot(dE, v3) < 0:
        dE = -dE
    angE = angle(dE, v3)
    el = time.time() - t0
    conv = a9 < 10
    print(f"  [{name}] a9(v3)={a9:8.3f} {'收敛' if conv else '不收敛'} | 内层末∠(E,v3)={angE:6.2f} "
          f"odr={odr:5.2f} t1n={t1n.real:+.3f}{t1n.imag:+.3f}j t2n={t2n.real:+.3f}{t2n.imag:+.3f}j ({el:.0f}s)", flush=True)
    return dict(name=name, a9=a9, conv=conv, t1=t1n, t2=t2n,
                pts=[[float(x), float(y), float(z)] for x, y, z in pts])


def save_bad(rec, tag=""):
    fn = os.path.join(OUT, f"bad_ws25_{tag}{rec['name']}.json")
    with open(fn, "w", encoding="utf-8") as f:
        json.dump(rec["pts"], f)
    print(f"  ==> 保存不收敛点集: {fn}  a9={rec['a9']:.3f}")
    return fn


# ---------- 合成点集生成器 ----------
def gen_sphere_vol(seed, n=200):
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n:
        p = rng.uniform(-1, 1, 3)
        if np.dot(p, p) <= 1.0:
            pts.append(p)
    return pts


def gen_sphere_shell(seed, n=200, R=1.2, sig=0.12):
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n:
        v = rng.normal(0, 1, 3)
        nrm = np.linalg.norm(v)
        if nrm < 1e-9:
            continue
        r = max(R + rng.normal(0, sig), 1e-3)
        pts.append(v / nrm * r)
    return pts


def gen_cube(seed, n=200):
    rng = np.random.default_rng(seed)
    return list(rng.uniform(-1, 1, (n, 3)))


def gen_ellipsoid(seed, n=200, radii=(1.3, 1.0, 0.8)):
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n:
        v = rng.normal(0, 1, 3)
        nrm = np.linalg.norm(v)
        if nrm < 1e-9:
            continue
        r = rng.uniform(0, 1) ** (1.0 / 3.0) * nrm  # 均匀球内再拉伸
        u = v / nrm
        p = np.array([u[0] * radii[0], u[1] * radii[1], u[2] * radii[2]]) * r * 1.2
        pts.append(p)
    return pts


def main():
    datasets = []
    # 真实 json
    for fn in ["pyjson.json", "points.json", "points1.json"]:
        path = os.path.join(RES, fn)
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw[0], dict):
            pts = [[p["x"], p["y"], p["z"]] for p in raw]
        else:
            pts = [list(p) for p in raw]
        datasets.append((fn.replace(".json", ""), pts))
    # 合成
    for s in range(6):
        datasets.append((f"sphere_vol_s{s}", gen_sphere_vol(s)))
        datasets.append((f"shell12_s{s}", gen_sphere_shell(s, R=1.2)))
        datasets.append((f"shell10_s{s}", gen_sphere_shell(s, R=1.0, sig=0.10)))
    for s in range(4):
        datasets.append((f"cube_s{s}", gen_cube(s)))
        datasets.append((f"ellip_s{s}", gen_ellipsoid(s)))
    print(f"扫描 {len(datasets)} 个点集, 配置 line81: axis=v3, (0,1)冷启, 50iter, wDir={W_DIR}:wSelf={W_SELF}")
    bad = []
    for name, pts in datasets:
        if len(pts) < 4:
            print(f"  [{name}] 点数不足, 跳过")
            continue
        rec = test_dataset(pts, name)
        if not rec["conv"]:
            bad.append(rec)
            save_bad(rec, "")
    print(f"\n==== 汇总 ====")
    print(f"共 {len(datasets)} 点集: 不收敛 {len(bad)} 个")
    for r in bad:
        print(f"  {r['name']}: a9={r['a9']:.2f}")
    # 汇总记录
    with open(os.path.join(OUT, "ws25_scan_summary.json"), "w", encoding="utf-8") as f:
        json.dump([{k: v for k, v in r.items() if k != "pts"} for r in bad], f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
