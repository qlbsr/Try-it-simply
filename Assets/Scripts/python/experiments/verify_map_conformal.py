"""
verify_map_conformal.py — 证明/否证 MapDoubleConeToLeaf 的共形性
================================================================================

用户给出的判据。设双锥局部坐标 (theta, s), 映射 (theta,s) -> (rho cos phi, rho sin phi),
曲面度规 g = diag(r^2(s), 1)。共形等价于拉回度规 f*delta = lambda^2 g, 即

  ① 正交:      F = rho_theta*rho_s + rho^2*phi_theta*phi_s = 0
  ② 各向同性:  E/G = r^2(s),   E = rho_theta^2 + rho^2 phi_theta^2
                                G = rho_s^2     + rho^2 phi_s^2

等价说法: 存在全纯 w(z), z = theta + i*tau(s), 使 phi + i*ln(rho) = w(z)。

本文件: 取双锥上按 (theta, s) 参数化的点, 过一遍 MapDoubleConeToLeaf 的真实实现
        (复用 experiments/verify_sjysjy_map.py 的全文复刻), 用中心差分求偏导,
        直接算 F/sqrt(EG) 与 E/G, 看 ① ② 是否成立。

运行: python verify_map_conformal.py
"""
import io, math, os, sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import verify_sjysjy_map as R          # th2 / dd2 / find_solutions / map_double_cone_to_leaf

D2 = 44.713528
RP = 1.0


def cone_point(theta, s, d2_deg=D2):
    """双锥上按 (theta, s) 参数化的点: 顶点在原点, 半顶角 d2, s 为沿母线的距离。
       半径 r(s) = s*sin(d2), 这与 th4 里 rs = r*sin(d2) 一致。"""
    a = math.radians(d2_deg)
    r = s * math.sin(a)
    return np.array([r * math.sin(theta), r * math.cos(theta), s * math.cos(a)])


def map_one(p, d2_deg=D2, rp=RP):
    """调用真实复刻实现, 取复平面坐标 (rho, phi)。"""
    try:
        out = R.map_double_cone_to_leaf([p], d2_deg, rp, verbose=False)
    except TypeError:
        out = R.map_double_cone_to_leaf([p], d2_deg, rp)
    uvs = out[0] if isinstance(out, tuple) else out
    uv = uvs[0]
    rho = math.hypot(uv[0], uv[1])
    phi = math.atan2(uv[1], uv[0])
    return rho, phi


def polar_derivs(theta, s, h=1e-5, d2_deg=D2):
    """中心差分求 rho, phi 对 theta 与 s 的偏导。"""
    p_tp = cone_point(theta + h, s, d2_deg)
    p_tm = cone_point(theta - h, s, d2_deg)
    p_sp = cone_point(theta, s + h, d2_deg)
    p_sm = cone_point(theta, s - h, d2_deg)
    rt_p, ph_tp = map_one(p_tp)
    rt_m, ph_tm = map_one(p_tm)
    rs_p, ph_sp = map_one(p_sp)
    rs_m, ph_sm = map_one(p_sm)
    rho_t = (rt_p - rt_m) / (2 * h)
    rho_s = (rs_p - rs_m) / (2 * h)

    def dwrap(a, b):
        dd = a - b
        while dd > math.pi:
            dd -= 2 * math.pi
        while dd < -math.pi:
            dd += 2 * math.pi
        return dd

    phi_t = dwrap(ph_tp, ph_tm) / (2 * h)
    phi_s = dwrap(ph_sp, ph_sm) / (2 * h)
    rho = 0.5 * (rt_p + rt_m)
    return rho, rho_t, rho_s, phi_t, phi_s


def test():
    print("=" * 78)
    print("MapDoubleConeToLeaf 共形性检验 (判据 ①②)")
    print("=" * 78)
    print(f"  d2 = {D2}°   rp = {RP}   双锥参数化 r(s) = s·sin(d2)")
    print()
    print("  ① 正交:      F/sqrt(E·G) 应 ≈ 0")
    print("  ② 各向同性:  E/G         应 ≈ r^2(s)  (只依赖 s)")
    print()
    print(f"  {'theta':>8} {'s':>7} {'rho':>12} {'E':>13} {'G':>13} "
          f"{'F/sqrt(EG)':>13} {'E/G':>13} {'r^2(s)':>12} {'E/G / r^2':>11}")
    rows = []
    for s in (0.5, 1.0, 2.0):
        for theta in (0.0, 1.0, 2.5, 4.0, 5.5):
            try:
                rho, rt, rs, pt, ps = polar_derivs(theta, s)
            except Exception as e:
                print(f"  {theta:>8.2f} {s:>7.2f}   [异常] {type(e).__name__}: {e}")
                continue
            E = rt * rt + rho * rho * pt * pt
            G = rs * rs + rho * rho * ps * ps
            F = rt * rs + rho * rho * pt * ps
            denom = math.sqrt(max(E * G, 1e-300))
            r2 = (s * math.sin(math.radians(D2))) ** 2
            ratio = (E / G) / r2 if G > 0 and r2 > 0 else float("nan")
            rows.append((theta, s, rho, E, G, F / denom, E / G, r2, ratio))
            print(f"  {theta:>8.2f} {s:>7.2f} {rho:>12.6f} {E:>13.6e} {G:>13.6e} "
                  f"{F/denom:>13.6f} {E/G:>13.6f} {r2:>12.6f} {ratio:>11.4f}")
    print()
    if not rows:
        print("  >>> 一行都没算出来: 复刻实现的调用接口需要核对。")
        return
    orth = max(abs(r[5]) for r in rows)
    iso_dev = max(abs(r[8] - 1.0) for r in rows if not math.isnan(r[8]))
    print("=" * 78)
    print("结论")
    print("=" * 78)
    print(f"  ① max |F/sqrt(EG)| = {orth:.6f}     (判据要求 ≈ 0)")
    print(f"  ② max |E/G / r^2(s) - 1| = {iso_dev:.6f}   (判据要求 ≈ 0)")
    print()
    if orth < 1e-3 and iso_dev < 1e-3:
        print("  => ①② 都成立: MapDoubleConeToLeaf 是【共形映射】, 数学意义成立。")
    elif orth < 1e-3:
        print("  => ①成立(保角)但②不成立: 是共形但与目标度规差一个正因子,")
        print("     即 f*delta = lambda^2 g 里的 lambda 不等于常数 —— 还需要标定 lambda。")
    else:
        print("  => ①不成立: 在当前参数化下【不是】共形映射。")
        print("     注意这可能是参数化 (theta,s)->三维点 的选择问题,")
        print("     而不是映射本身的问题; 需要先确认双锥上 (theta,s) 的正确定义。")
    print()
    print("  说明: 我用的 (theta,s) 参数化是 r(s)=s·sin(d2)、p=(r sinθ, r cosθ, s cos d2),")
    print("        依据是 th4 的 rs = r·sin(d2)。若你定义的双锥坐标不同, 请指出, 我改参数化重跑。")


if __name__ == "__main__":
    test()
