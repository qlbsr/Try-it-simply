# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
三角形基本性质探索 (用户方向): 边长 / 内角和 / 质心 / 不变量
  三角形 △(p0, v·c, -v·c):
    a = |p0 - v·c|,  b = |p0 + v·c|,  底边 = 2c (恒定!)
  基本性质:
    T1: a² + b² = 2(|p0|² + c²)  [与 v 无关! 三角形族不变量]
    T2: 质心 G = (p0 + v·c + (-v·c))/3 = p0/3  [与 v 无关!]
    T3: 内角 (余弦定理) ∠(p0) 对边 2c, ∠(vc) 对边 b, ∠(-vc) 对边 a
    T4: 面积 S = (1/2)·底·高, 高 = p0 到 v 轴距离
    T5: 半周长 s = (a+b+2c)/2, 海伦公式
  目标: 找 "d 与 PCA 特殊关系 (|Δ|≤16)" 时三角形性质的规律
"""
import json
import math

import numpy as np

import data_driven_axis as dd
import n2sjy2 as n2
import nsjy_algorithms as m

PYJSON = r"C:\Users\23128\My project (2)\Assets\Resources\pyjson.json"


def angle(u, v):
    return math.degrees(math.acos(np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1)))


def triangle_props(p0, v, c):
    """三角形 △(p0, v·c, -v·c) 的全部基本性质"""
    v = np.asarray(v, float) / np.linalg.norm(v)
    A = np.asarray(p0, float)
    B = v * c
    C = -v * c
    a = np.linalg.norm(A - B)      # 对边 = 底? 约定: a=|p0-vc|, b=|p0+vc|
    b = np.linalg.norm(A - C)
    base = 2 * c
    # 内角 (顶点处)
    # 顶点 A=p0 处夹角 (边 ab 夹 base)
    cosA = (a * a + b * b - base * base) / (2 * a * b)
    angA = math.degrees(math.acos(np.clip(cosA, -1, 1)))
    # 顶点 B=vc 处 (边 a, base 夹 b)
    cosB = (a * a + base * base - b * b) / (2 * a * base)
    angB = math.degrees(math.acos(np.clip(cosB, -1, 1)))
    angC = 180 - angA - angB
    # 质心 (三个顶点平均)
    G = (A + B + C) / 3
    # 面积 (海伦)
    s_ = (a + b + base) / 2
    area = math.sqrt(max(0.0, s_ * (s_ - a) * (s_ - b) * (s_ - base)))
    # p0 到 v 轴的距离 (高)
    h = np.linalg.norm(np.cross(A, v))   # |A|·sin∠(A,v)
    return dict(a=a, b=b, base=base, angA=angA, angB=angB, angC=angC,
                sum_ang=angA + angB + angC, G=G, area=area, h=h,
                a2b2=a * a + b * b, r0=np.linalg.norm(A))


def main():
    with open(PYJSON, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    pts = [np.array([p["x"], p["y"], p["z"]], float) for p in raw]
    P = np.array(pts, float)
    p0 = P[0]
    rp = n2.compute_rp(pts)
    d2 = math.asin(math.cos(math.radians(30)) / math.pi)
    a0 = rp * (1 + math.sin(d2))
    e3 = math.cos(d2) * 2 / (1 + math.sin(d2))
    c = rp * math.cos(d2) * e3
    _, r45, r30, _ = n2.yzqx(pts, rp)
    t1t, t2t = n2.compute_taus()
    t_ = complex(0, 1)
    pt, p1, p2, sig = dd.fast_probs(r30, r45, t1t, t2t, a0)
    F1, F2 = m.extract_foci(pts, p1, sig[1], p2, sig[2])
    f1, f2 = m.extract_foci(pts, pt, sig[1], pt, sig[2])
    pt01, p101, p201, sig01 = dd.fast_probs(r30, r45, t_, t_, a0)
    F01, F02 = m.extract_foci(pts, p101, sig01[1], p201, sig01[2])
    f01, f02 = m.extract_foci(pts, pt01, sig01[1], pt01, sig01[2])
    v = m.pca(pts)[1]
    v = v / np.linalg.norm(v)
    v1 = (np.array(F1, float) - np.array(F2, float)); v1 = v1 / np.linalg.norm(v1)
    v2 = (np.array(f1, float) - np.array(f2, float)); v2 = v2 / np.linalg.norm(v2)
    v3 = (np.array(F01, float) - np.array(F02, float)); v3 = v3 / np.linalg.norm(v3)
    v4 = (np.array(f01, float) - np.array(f02, float)); v4 = v4 / np.linalg.norm(v4)

    dirs = {"v(pca)": v, "v1": v1, "v2": v2, "v3": v3, "v4": v4}
    print("=" * 100)
    print("三角形基本性质: △(p0, v·c, -v·c)   p0=%s |p0|=%.3f  c=%.3f" % (np.round(p0, 3), np.linalg.norm(p0), c))
    print("=" * 100)
    print(f"{'dir':8s} {'a':>6s} {'b':>6s} {'底2c':>6s} {'a²+b²':>8s} {'∠A(p0)':>7s} "
          f"{'∠B':>6s} {'∠C':>6s} {'内角和':>6s} {'质心x':>6s} {'质心y':>6s} {'质心z':>6s} {'高h':>6s}")
    for nm, vv in dirs.items():
        t = triangle_props(p0, vv, c)
        print(f"{nm:8s} {t['a']:6.3f} {t['b']:6.3f} {t['base']:6.3f} {t['a2b2']:8.3f} "
              f"{t['angA']:7.2f} {t['angB']:6.2f} {t['angC']:6.2f} {t['sum_ang']:6.2f} "
              f"{t['G'][0]:6.3f} {t['G'][1]:6.3f} {t['G'][2]:6.3f} {t['h']:6.3f}")
    print()
    print("检查:")
    print("  T1 a²+b² 恒定? (应 = 2(|p0|²+c²) = %.3f)" % (2 * (np.linalg.norm(p0) ** 2 + c ** 2)))
    print("  T2 质心 = p0/3 =", np.round(p0 / 3, 3))
    print("  T3 内角和 = 180?")
    print()
    print("===> 若 T1/T2 恒定, 则三角形族不变量与 v 无关;")
    print("     差异只在 ∠A(顶点p0处) / 高 h / a·b 等含 v 的量")
    print("DONE")


if __name__ == "__main__":
    main()
