# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
三角形方法完整版: 用 3 个参考方向 (v1, v3, v5 反演的 p0 视角) 定位 v5
  消除双解: s → θ ∈ {θ0, 180-θ0}; 用球面三角把 v5 表示在 p0̂ 锥面上
  锥面参数化: v5 = cos(θ5)·p0̂ + sin(θ5)·(cos(φ)·e1 + sin(φ)·e2)
  约束: ∠(v5, v1) 与 ∠(v5, v3) 已知 (从向量直接算, 无需 PCA)
  → 解 φ, 得到 v5 的方向余弦 → 完全脱离 PCA 重建当前方向
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


def dist(p, q):
    return np.linalg.norm(np.asarray(p, float) - np.asarray(q, float))


def build_inv_table(p0, c, a, n=3601):
    r0 = np.linalg.norm(p0)
    th = np.linspace(0, 180, n)
    s = np.zeros(n)
    for i, t in enumerate(th):
        rad = math.radians(t)
        d1 = math.sqrt(r0 ** 2 + c ** 2 - 2 * r0 * c * math.cos(rad))
        d2 = math.sqrt(r0 ** 2 + c ** 2 + 2 * r0 * c * math.cos(rad))
        s[i] = math.exp(-abs(d1 + d2 - 2 * a) / (2 * a))
    return th, s


def invert_candidates(s_val, th_grid, s_grid, tol=1e-3):
    """s → θ 全部候选 (处理双解)"""
    near = np.where(np.abs(s_grid - s_val) < tol)[0]
    if len(near) == 0:
        i = np.argmin(np.abs(s_grid - s_val))
        near = [i]
    ths = th_grid[near]
    # 聚类成两个候选 (0-90 和 90-180)
    cands = []
    for base in (0, 90):
        m_ = ths[(ths >= base) & (ths < base + 90)]
        if len(m_):
            cands.append(float(np.median(m_)))
    return cands


def main():
    with open(PYJSON, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    pts = [np.array([p["x"], p["y"], p["z"]], float) for p in raw]
    P = np.array(pts, float)
    p0 = P[0]
    rp = n2.compute_rp(pts)
    d2 = math.asin(math.cos(math.radians(30)) / math.pi)
    a = rp * (1 + math.sin(d2))
    e3 = math.cos(d2) * 2 / (1 + math.sin(d2))
    c = rp * math.cos(d2) * e3
    _, r45, r30, _ = n2.yzqx(pts, rp)
    t1t, t2t = n2.compute_taus()
    t_ = complex(0, 1)
    pt, p1, p2, sig = dd.fast_probs(r30, r45, t1t, t2t, a)
    F1, F2 = m.extract_foci(pts, p1, sig[1], p2, sig[2])
    pt01, p101, p201, sig01 = dd.fast_probs(r30, r45, t_, t_, a)
    F01, F02 = m.extract_foci(pts, p101, sig01[1], p201, sig01[2])
    v = m.pca(pts)[1]
    v = v / np.linalg.norm(v)
    v1 = (np.array(F1, float) - np.array(F2, float))
    v1 = v1 / np.linalg.norm(v1)
    v3 = (np.array(F01, float) - np.array(F02, float))
    v3 = v3 / np.linalg.norm(v3)

    th_grid, s_grid = build_inv_table(p0, c, a)

    def s_of(vv):
        d1 = dist(p0, vv * c)
        d2 = dist(p0, -vv * c)
        return math.exp(-abs(d1 + d2 - 2 * a) / (2 * a))

    # 参考方向已知夹角 (无需 PCA)
    a13 = angle(v1, v3)
    p0hat = p0 / np.linalg.norm(p0)

    # 构造 p0̂ 的正交基
    tmp = np.array([1.0, 0, 0]) if abs(p0hat[0]) < 0.9 else np.array([0.0, 1, 0])
    e1 = np.cross(p0hat, tmp)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(p0hat, e1)

    print(f"∠(v1,v3)={a13:.1f}°  ∠(p0̂,v1)={angle(p0hat, v1):.1f}°  ∠(p0̂,v3)={angle(p0hat, v3):.1f}°")

    # 用反演的 θ 测试: 对每个参考方向, 反演得 θ, 重建该方向, 对比真值
    print()
    print("=== 单参考反演重建 (双解选对时) ===")
    for name, vv in [("v1", v1), ("v3", v3), ("v5", v * 0.5 + v1 * 0.5)]:
        vv = vv / np.linalg.norm(vv)
        s_val = s_of(vv)
        cands = invert_candidates(s_val, th_grid, s_grid)
        th_true = angle(p0hat, vv)
        # 用真值θ直接重建方向需要方位角; 这里只验证θ反演
        ok = any(abs(c - th_true) < 1.5 or abs(c - (180 - th_true)) < 1.5 for c in cands)
        print(f"  {name}: s={s_val:.5f} 真θ={th_true:6.1f}° 候选={[f'{c:.1f}' for c in cands]} "
              f"{'✓(含真值)' if ok else '✗'}")

    print()
    print("=== 双参考定位 v5 (已知 ∠(v5,v1), ∠(v5,v3) + 反演 θ5) ===")
    # 构造一个测试 v5 (在 v1 与 v3 之间)
    v5_true = v1 * math.cos(math.radians(40)) + v3 * math.sin(math.radians(40))
    v5_true = v5_true / np.linalg.norm(v5_true)
    s5 = s_of(v5_true)
    cands5 = invert_candidates(s5, th_grid, s_grid)
    print(f"  测试 v5: 真方向 ∠(p0̂,v5)={angle(p0hat, v5_true):.1f}°  "
          f"s5={s5:.5f}  θ候选={[f'{c:.1f}' for c in cands5]}")
    # 已知: ∠(v5,v1), ∠(v5,v3) (从向量, 无 PCA)
    a51 = angle(v5_true, v1)
    a53 = angle(v5_true, v3)
    print(f"  ∠(v5,v1)={a51:.1f}°  ∠(v5,v3)={a53:.1f}°")
    # 锥面参数化求解: v5 = cos(θ)p0̂ + sin(θ)(cosφ e1 + sinφ e2)
    # ∠(v5,v1)=a51 → cos(a51) = v5·v1 (线性的 cosφ)
    # 对每个 θ 候选, 解 φ
    for th_cand in cands5:
        th = math.radians(th_cand)
        # v5·v1 = cosθ(p0̂·v1) + sinθ(cosφ·(e1·v1) + sinφ·(e2·v1))
        A = math.cos(th) * np.dot(p0hat, v1)
        B = math.sin(th) * np.dot(e1, v1)
        C = math.sin(th) * np.dot(e2, v1)
        # 需要解: A + B cosφ + C sinφ = cos(a51)
        # B cosφ + C sinφ = cos(a51) - A = R
        R = math.cos(math.radians(a51)) - A
        # 归一化 (B,C): r = sqrt(B²+C²), B'=B/r, C'=C/r → r(B'cosφ+C'sinφ)=R
        r = math.hypot(B, C)
        if r < 1e-9:
            print(f"    θ={th_cand:.1f}°: 退化 (B=C=0), 跳过")
            continue
        Bp, Cp = B / r, C / r
        # Bp cosφ + Cp sinφ = R/r → cos(φ - φ0) = R/(r·?) 
        # Bp cosφ + Cp sinφ = cos(φ)·Bp + sin(φ)·Cp = R/r
        # 令 φ0 = atan2(Cp, Bp): cos(φ-φ0) = R/r
        phi0 = math.atan2(Cp, Bp)
        arg = R / r
        if abs(arg) > 1:
            print(f"    θ={th_cand:.1f}°: 无解 (arg={arg:.2f})")
            continue
        for sgn in (1, -1):
            phi = phi0 + sgn * math.acos(max(-1, min(1, arg)))
            v5_recon = math.cos(th) * p0hat + math.sin(th) * (
                math.cos(phi) * e1 + math.sin(phi) * e2)
            # 验证 ∠(v5_recon, v3)
            chk53 = angle(v5_recon, v3)
            err = abs(chk53 - a53)
            print(f"    θ={th_cand:5.1f}° φ={math.degrees(phi):7.1f}° → "
                  f"重建∠(v5,v3)={chk53:6.1f}° (真值{a53:.1f}°, 误差{err:.1f}°) "
                  f"{'✓' if err < 3 else ''}")
    print()
    print("注: 双参考 (v1,v3) 已足够; 若误差<3° 则三角定位成功, 完全无 PCA")
    print("DONE")


if __name__ == "__main__":
    main()
