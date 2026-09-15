# -*- coding: utf-8 -*-
"""
sjy 叶映射 v2 —— 完整修正版 (含 NaN 防护 + 有效性统计)

链条:
  1. 大锥几何(隧道):  h = rp·cosα,  R = rp·sinα,  H = πR
  2. 卡住判定 theta2:  坐标规范化(π/2) / 带残差检查的真根 / 近似 atan2(h/2,H)
  3. 逐点:  th2(theta2, Y) -> 通道宽 h1 与方向 u2;  旋转 q;  环心 C
  4. dd2:   三角形 -> 马丁椭圆 (周长 c3, 半轴 a,b, 中心, 平面坐标)
  5. 参数角 t = atan2((v-cy)/b, (u-cx)/a)     有符号, 覆盖整圈
  6. 椭圆展开: k = √(1-(b/a)²) -> K, F(t,k) -> theta = F/(4K) ∈ [0,1)
  7. 偏移参数: ratio -> x -> s -> sec²         (拉伸锐度)
  8. 体积:  pappus(旋转体) / pipe(柱体) / ellipsoid(拼三轴)
  9. 复平面点: uvs = rad · e^{i·2π·theta}

与 v1 的差别:
  - 删掉无意义的 vector + FromToRotation + Quaternion.Angle 相位 (方位信息恒丢失)
  - 相位改成椭圆参数角 t -> 椭圆积分 theta (有符号, 整圈, 连续)
  - 体积第三维改为几何同源的量 (Pappus 旋转半径 / 路径长度), 不再方向重复
  - 参考体积 Vc 量纲修正为 [L^3]
  - 全程防护: 每个可能 NaN 的环节都 clamp/fallback 并计数
"""
import json
import math
import os
import sys

import numpy as np
from scipy.special import ellipe, ellipk, ellipkinc

PI = math.pi


# ==================================================================
# 椭圆展开工具 (k=0 退化为标准圆)
# ==================================================================
def elliptic_K(k):
    """第一类完全椭圆积分 K(k), 参数为模数 k"""
    m = min(max(k * k, 0.0), 1.0 - 1e-15)
    return float(ellipk(m))


def elliptic_F(phi, k):
    """第一类不完全椭圆积分 F(phi,k), 支持任意 phi (按周期展开)"""
    m = min(max(k * k, 0.0), 1.0 - 1e-15)
    K = float(ellipk(m))
    # 折到 [-pi, pi]
    p = phi % (2 * PI)
    if p > PI:
        p -= 2 * PI
    s = 1.0 if p >= 0 else -1.0
    a = abs(p)
    if a <= PI / 2:
        return s * float(ellipkinc(a, m))
    return s * (2.0 * K - float(ellipkinc(PI - a, m)))


# ==================================================================
# 几何工具
# ==================================================================
def th2(theta, H):
    """两条母线端点; y 分量恒 = H (高度=环周长刻度)"""
    s, c = math.sin(theta), math.cos(theta)
    if abs(s) < 1e-300:
        s = 1e-300 if s >= 0 else -1e-300
    tanX = -H / s
    c2 = np.array([math.atan(tanX), H, tanX * c])
    tanX_low = H / s
    c1 = np.array([math.atan(tanX_low), H, tanX_low * c])
    return c1, c2


def from_to_rotation(a, b):
    """Rodrigues: 把 a 转到 b 的旋转矩阵 (对应 Unity FromToRotation)"""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-300 or nb < 1e-300:
        return np.eye(3)
    a = a / na
    b = b / nb
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        perp = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
        ax = np.cross(a, perp)
        ax /= (np.linalg.norm(ax) + 1e-300)
        Kx = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        return np.eye(3) + 2 * Kx @ Kx
    Kx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + Kx + Kx @ Kx * ((1 - c) / (s * s))


def dd2_full(A, B, C):
    """C# dd2 忠实 + 额外返回平面坐标

    返回 (C_ell, a_ell, b_ell, center, pC, flags)
    flags: 'degenerate' 三点共线, 'bpatched' 短轴补丁被触发
    """
    flags = []
    n = np.cross(B - A, C - A)
    nn = float(np.linalg.norm(n))
    if nn < 1e-12:
        flags.append("degenerate")
        return 0.0, 0.0, 0.0, np.zeros(2), np.zeros(2), flags
    n = n / nn
    dCA = C - A
    nCA = float(np.linalg.norm(dCA))
    if nCA < 1e-12:
        flags.append("degenerate")
        return 0.0, 0.0, 0.0, np.zeros(2), np.zeros(2), flags
    e1 = dCA / nCA
    e2 = np.cross(n, e1)
    e2 /= (np.linalg.norm(e2) + 1e-300)
    A2 = np.zeros(2)
    B2 = np.array([np.dot(B - A, e1), np.dot(B - A, e2)])
    C2 = np.array([np.dot(C - A, e1), np.dot(C - A, e2)])
    center = (A2 + B2 + C2) / 3.0
    pts = [A2, B2, C2]
    sxx = sum((p[0] - center[0]) ** 2 for p in pts)
    syy = sum((p[1] - center[1]) ** 2 for p in pts)
    sxy = sum((p[0] - center[0]) * (p[1] - center[1]) for p in pts)
    tr = sxx + syy
    det = sxx * syy - sxy * sxy
    disc = math.sqrt(max(0.0, tr * tr - 4 * det))
    l1 = (tr + disc) / 2
    l2 = (tr - disc) / 2
    ae = math.sqrt(max(0.0, l1 / 2))
    be = math.sqrt(max(0.0, l2 / 2))
    if be < 1e-6:
        be = ae * 0.1
        flags.append("bpatched")
    Ce = PI * (3 * (ae + be) - math.sqrt(max(0.0, (3 * ae + be) * (ae + 3 * be))))
    return Ce, ae, be, center, C2, flags


# ==================================================================
# 叶映射 v2
# ==================================================================
class LeafMapV2:
    """一次完整映射: 点集 -> 复平面点集 (uvs)

    theta2_mode:  'spec'    -> π/2 (坐标规范化, 洞口对齐 x 轴, 用户设计)
                  'approx'  -> atan2(h/2, H)
    scheme:       'pappus'  -> 环体积 (子午面椭圆绕轴旋转, 与锥体积同类型)
                  'pipe'    -> 管道柱体 (椭圆面 x 路径长)
                  'ellipsoid'-> 拼三轴椭球 (保留对比, 不推荐)
    rad_pow:      模长 = (V/Vc)^rad_pow
    """

    def __init__(self, rp, d2_deg, theta2_mode="spec", scheme="pappus",
                 rad_pow=1.0, volume_floor=1e-12):
        self.rp = float(rp)
        self.d2_deg = float(d2_deg)
        self.alpha = math.radians(self.d2_deg)
        self.h = self.rp * math.cos(self.alpha)
        self.R = self.rp * math.sin(self.alpha)
        self.H = PI * self.R
        if theta2_mode == "spec":
            self.theta2 = PI / 2
        else:
            self.theta2 = math.atan2(0.5 * self.h, self.H) if self.H > 0 else PI / 2
        self.scheme = scheme
        self.rad_pow = rad_pow
        self.volume_floor = volume_floor
        # 轴向 (母线差, 用户设计: 洞口对齐)
        c1, c2 = th2(self.theta2, self.H if self.H > 0 else 1e-12)
        d = c1 - c2
        nd = np.linalg.norm(d)
        self.v3 = d / nd if nd > 1e-12 else np.array([1.0, 0, 0])
        # 参考体积 (相似律, [L^3])
        self.V0 = (1.0 / 3.0) * PI * self.R * self.R * self.h     # 全锥体积

    # --------------------------------------------------------------
    def map_point(self, P):
        """单点映射; 返回 (uvs_complex, diag)"""
        diag = dict(flags=[], h1=math.nan, c3=math.nan, a=math.nan, b=math.nan,
                    k=math.nan, theta=math.nan, t=math.nan,
                    ratio1=math.nan, ratio2=math.nan, s=math.nan, s2=math.nan,
                    V=math.nan, Vc=math.nan, rad=math.nan, rho_c=math.nan)
        P = np.asarray(P, float)
        Y = float(P[1])
        H = self.H if self.H > 1e-12 else 1e-12

        # ---- 3. th2: 通道宽与方向 ----
        v0, v1 = th2(self.theta2, Y)
        h1 = float(np.linalg.norm(v0 - v1))
        diag["h1"] = h1
        if h1 < 1e-12:
            diag["flags"].append("h1_zero")
            h1 = 1e-12
        u2 = (v0 - v1) / (np.linalg.norm(v0 - v1) + 1e-300)
        M = from_to_rotation(self.v3, u2)
        Prot = M @ P
        y_norm = h1 / (self.h if abs(self.h) > 1e-12 else 1e-12)
        Cc = np.array([0.0, y_norm * H, 0.0])

        # ---- 4. dd2: 马丁椭圆 ----
        c3, ae, be, center, pC, fl = dd2_full(np.zeros(3), Cc, Prot)
        diag["flags"].extend(fl)
        diag["c3"], diag["a"], diag["b"] = c3, ae, be
        if "degenerate" in fl or ae < 1e-12:
            diag["flags"].append("ellipse_degenerate")
            return complex(math.nan, math.nan), diag

        # ---- 5. 参数角 t (有符号, 覆盖整圈) ----
        du = (pC[0] - center[0]) / max(ae, 1e-12)
        dv = (pC[1] - center[1]) / max(be, 1e-12)
        t = math.atan2(dv, du)
        diag["t"] = t

        # ---- 6. 椭圆展开: k -> K -> F -> theta ∈ [0,1) ----
        ratio_ab = be / ae
        k2 = 1.0 - ratio_ab * ratio_ab
        if k2 < 0.0:                       # 理论上不会 (ae>=be), 防御
            k2 = 0.0
            diag["flags"].append("k2_clamped")
        k = math.sqrt(min(k2, 1.0 - 1e-12))
        if k2 > 1.0 - 1e-12:
            diag["flags"].append("k_near1")
        K = elliptic_K(k)
        diag["k"] = k
        if not (K > 1e-12):
            diag["flags"].append("K_zero")
            return complex(math.nan, math.nan), diag
        u_ell = elliptic_F(t, k)
        theta = (u_ell / (4.0 * K)) % 1.0
        diag["theta"] = theta

        # ---- 7. 偏移参数 (拉伸锐度) ----
        c2d = float(np.linalg.norm(Prot))
        Hcone = float(np.linalg.norm(Cc))
        r1 = c3 / c2d if c2d > 1e-12 else math.nan
        r2 = c3 / Hcone if Hcone > 1e-12 else math.nan
        diag["ratio1"], diag["ratio2"] = r1, r2
        r1c = max(r1, 1.0) if math.isfinite(r1) else 1.0
        r2c = max(r2, 1.0) if math.isfinite(r2) else 1.0
        if math.isfinite(r1) and r1 < 1.0:
            diag["flags"].append("ratio1_clamped")
        if math.isfinite(r2) and r2 < 1.0:
            diag["flags"].append("ratio2_clamped")
        x1 = math.acos(1.0 / r1c) if r1c > 1e-300 else 0.0
        x2 = math.acos(1.0 / r2c) if r2c > 1e-300 else 0.0
        s = 1.0 - 2.0 * x1 / PI
        s2 = 1.0 - 2.0 * x2 / PI
        diag["s"], diag["s2"] = s, s2

        # ---- 8. 体积 ----
        Vc = (y_norm ** 3) * self.V0 if abs(y_norm) > 1e-12 else 0.0
        diag["Vc"] = Vc
        rho_c = abs(center[1])                                  # 椭圆中心到轴距离
        diag["rho_c"] = rho_c
        A_ell = PI * ae * be
        if self.scheme == "pappus":
            V = A_ell * 2.0 * PI * rho_c
        elif self.scheme == "pipe":
            V = A_ell * c3
        else:                                                   # ellipsoid
            wrap1 = math.sqrt(max(0.0, r1c * r1c - 1.0))
            wrap2 = math.sqrt(max(0.0, r2c * r2c - 1.0))
            c_axis = abs(h1 * wrap2 - h1 * wrap1) / (2.0 * PI)
            V = (4.0 / 3.0) * PI * ae * be * c_axis
        diag["V"] = V
        if not (math.isfinite(V)) or V <= 0.0:
            diag["flags"].append("V_zero")
            V = self.volume_floor
        if not (math.isfinite(Vc)) or Vc <= self.volume_floor:
            diag["flags"].append("Vc_zero")
            Vc = self.volume_floor
        rad = (V / Vc) ** self.rad_pow
        if not math.isfinite(rad):
            diag["flags"].append("rad_nonfinite")
            rad = 0.0
        diag["rad"] = rad

        # ---- 9. 复平面点 ----
        z = complex(rad * math.cos(2 * PI * theta), rad * math.sin(2 * PI * theta))
        if not (math.isfinite(z.real) and math.isfinite(z.imag)):
            diag["flags"].append("final_nan")
        return z, diag

    # --------------------------------------------------------------
    def map_all(self, points):
        uvs = []
        diags = []
        for P in points:
            z, dg = self.map_point(P)
            uvs.append(z)
            diags.append(dg)
        return np.array(uvs, complex), diags


# ==================================================================
# 测试
# ==================================================================
def run_test(points, name, d2_list, scheme="pappus"):
    rows = []
    for d2d in d2_list:
        lm = LeafMapV2(rp=float(np.mean(np.linalg.norm(points, axis=1))),
                       d2_deg=d2d, theta2_mode="spec", scheme=scheme)
        uvs, diags = lm.map_all(points)
        n = len(uvs)
        nan_cnt = int(np.sum(~np.isfinite(uvs)))
        flag_cnt = {}
        for dg in diags:
            for f in dg["flags"]:
                flag_cnt[f] = flag_cnt.get(f, 0) + 1
        thetas = np.array([dg["theta"] for dg in diags])
        rads = np.array([dg["rad"] for dg in diags])
        ks = np.array([dg["k"] for dg in diags])
        rows.append(dict(d2=d2d, n=n, nan=nan_cnt, flags=flag_cnt,
                         th_min=np.nanmin(thetas), th_max=np.nanmax(thetas),
                         th_span=(np.nanmax(thetas) - np.nanmin(thetas)),
                         rad_min=np.nanmin(rads), rad_max=np.nanmax(rads),
                         rad_med=np.nanmedian(rads),
                         k_min=np.nanmin(ks), k_max=np.nanmax(ks)))
    return rows


def main():
    RES = r"C:\Users\23128\My project (2)\Assets\Resources"
    sets = []
    for fn in ["points.json", "points1.json", "pyjson.json"]:
        path = os.path.join(RES, fn)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw[0], dict):
            P = np.array([[p["x"], p["y"], p["z"]] for p in raw], float)
        else:
            P = np.array([list(p) for p in raw], float)
        sets.append((fn.replace(".json", ""), P))

    d2_list = [5, 15, 25, 35, 45, 55, 65, 75, 85]

    for scheme in ["pappus", "pipe", "ellipsoid"]:
        print("=" * 100)
        print(f"体积方案: {scheme}")
        print("=" * 100)
        for name, P in sets:
            print(f"\n--- 点集 {name} (n={len(P)}) ---")
            print(f"  {'d2':>4s} {'NaN/总数':>10s} {'有效%':>7s} {'θ范围':>16s} {'θ跨度':>7s} "
                  f"{'rad中位':>9s} {'rad范围':>18s} {'k范围':>14s} {'防护触发':>28s}")
            rows = run_test(P, name, d2_list, scheme)
            for r in rows:
                eff = 100.0 * (r["n"] - r["nan"]) / r["n"]
                fl = ",".join(f"{k}x{v}" for k, v in sorted(r["flags"].items())) or "-"
                print(f"  {r['d2']:4d} {r['nan']:4d}/{r['n']:<5d} {eff:6.2f}% "
                      f"[{r['th_min']:.4f},{r['th_max']:.4f}] {r['th_span']:7.4f} "
                      f"{r['rad_med']:9.4f} [{r['rad_min']:.4g},{r['rad_max']:.4g}] "
                      f"[{r['k_min']:.4f},{r['k_max']:.4f}] {fl:>28s}")

    print("\n" + "=" * 100)
    print("判读标准")
    print("  · NaN/总数 = 0 且 有效% = 100%  → 映射全程无 NaN 堆积")
    print("  · θ跨度 接近 1.0 → 环坐标覆盖整圈 (v1 的相位只有半圈且跳变)")
    print("  · k 范围 反映该点集在锥面上的'偏离圆'程度 (k=0 为交接环标准圆)")
    print("  · 防护触发次数为 0 或极少 → 约束在有效域内, 未依赖补丁")


if __name__ == "__main__":
    main()
