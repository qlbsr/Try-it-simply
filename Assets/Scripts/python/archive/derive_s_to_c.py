# -*- coding: utf-8 -*-
"""
逐步推导: ratio -> x -> s -> sec^2 -> tan -> 卷绕半径 -> 管道厚度 c
用 points.json 真实数据走一遍每一步的中间值
"""
import json
import math

import numpy as np

from dd2_path_test import dd2, from_to_rotation, th2

PI = math.pi


def main():
    with open(r"C:\Users\23128\My project (2)\Assets\Resources\points.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    P = np.array([np.array(p, float) for p in raw], float)
    rp = float(np.mean(np.linalg.norm(P, axis=1)))

    d2d = 15.0
    alpha = math.radians(d2d)
    h = rp * math.cos(alpha)
    R = rp * math.sin(alpha)
    H = PI * R
    theta2 = PI / 2
    c1v, c2v = th2(theta2, H)
    v3 = (c1v - c2v) / np.linalg.norm(c1v - c2v)

    print(f"=== 设定: d2={d2d}°  h={h:.4f} R={R:.4f} H=πR={H:.4f} ===")

    for i in [0, 40, 120]:
        Pi = P[i]
        Y = float(Pi[1])
        v0, v1 = th2(theta2, Y)
        h1 = float(np.linalg.norm(v0 - v1))
        u2 = (v0 - v1) / (np.linalg.norm(v0 - v1) + 1e-30)
        M = from_to_rotation(v3, u2)
        Prot = M @ Pi
        y_norm = h1 / h
        Cc = np.array([0.0, y_norm * H, 0.0])
        c3, ae, be = dd2(np.zeros(3), Cc, Prot)
        c2 = float(np.linalg.norm(Prot))
        Hcone = float(np.linalg.norm(Cc))

        print(f"\n---------- 点 idx={i} ----------")
        print(f"  [0] 原始量: 椭圆周长 c3={c3:.6f}  点到原点 c2={c2:.6f}  "
              f"环心距 H_cone={Hcone:.6f}  通道宽 h1={h1:.6f}")

        ratio1 = c3 / c2
        ratio2 = c3 / Hcone
        print(f"  [1] 两个比值: ratio1=c3/c2={ratio1:.6f}   ratio2=c3/H_cone={ratio2:.6f}")

        x1 = math.acos(1.0 / math.sqrt(ratio1))
        x2 = math.acos(1.0 / math.sqrt(ratio2))
        print(f"  [2] 角: x1=acos(1/√ratio1)={x1:.6f} rad = {math.degrees(x1):.4f}°")
        print(f"           x2=acos(1/√ratio2)={x2:.6f} rad = {math.degrees(x2):.4f}°")
        print(f"      恒等式校验 sec²x1={1/math.cos(x1)**2:.6f} (应=ratio1={ratio1:.6f})")
        print(f"      恒等式校验 sec²x2={1/math.cos(x2)**2:.6f} (应=ratio2={ratio2:.6f})")

        s = 1.0 - 2.0 * x1 / PI
        s2 = 1.0 - 2.0 * x2 / PI
        print(f"  [3] 偏移参数: s=1-2x1/π={s:.6f}   s2=1-2x2/π={s2:.6f}")

        th_s = s * PI / 2
        th_s2 = s2 * PI / 2
        print(f"  [4] theta_s=s·π/2={th_s:.6f} rad   (校验 = π/2-x1 = {PI/2-x1:.6f})")
        print(f"      theta_s2=s2·π/2={th_s2:.6f} rad  (校验 = π/2-x2 = {PI/2-x2:.6f})")

        sec2_s = 1.0 + math.tan(th_s) ** 2
        sec2_s2 = 1.0 + math.tan(th_s2) ** 2
        print(f"  [5] sec²_s = 1+tan²(theta_s) = {sec2_s:.6f}   "
              f"(校验 ratio1/(ratio1-1) = {ratio1/(ratio1-1):.6f})")
        print(f"      sec²_s2 = {sec2_s2:.6f}   "
              f"(校验 ratio2/(ratio2-1) = {ratio2/(ratio2-1):.6f})")

        tx1 = math.tan(x1)
        tx2 = math.tan(x2)
        print(f"  [6] 卷绕系数: tan(x1)=√(ratio1-1)={tx1:.6f}  (直接√校验={math.sqrt(ratio1-1):.6f})")
        print(f"                tan(x2)=√(ratio2-1)={tx2:.6f}  (直接√校验={math.sqrt(ratio2-1):.6f})")
        print(f"      also: cot(theta_s)={1/math.tan(th_s):.6f}  cot(theta_s2)={1/math.tan(th_s2):.6f}")

        D = h1
        n = 1.0
        rho_in = D * tx1 / (2 * PI * n)
        rho_out = D * tx2 / (2 * PI * n)
        c_axis = abs(rho_out - rho_in)
        print(f"  [7] 卷绕半径 (D=h1={D:.4f}, n={n}): ρ1={rho_in:.6f}  ρ2={rho_out:.6f}")
        print(f"  [8] 管道厚度 c = |ρ2-ρ1| = {c_axis:.6f}   [L] ✓")
        print(f"  [9] 体积 = (4/3)π·a·b·c = {(4/3)*PI*ae*be*c_axis:.8f}  (a={ae:.5f}, b={be:.5f})")

        # 对照: 用 h1/2 与 L_y 作第三轴 (方向重复)
        print(f"  [对照] 用 h1/2 作第三轴: 体积={(4/3)*PI*ae*be*(h1/2):.8f}")
        print(f"  [对照] 用 L_y=y·h={y_norm*h:.4f} 作第三轴: 体积={(4/3)*PI*ae*be*(y_norm*h):.8f}")

    print("\n=== 线性 vs 平方 的差别 (ratio 应被当作 sec 还是 sec²) ===")
    for r in [1.5, 2.5, 4.0]:
        x_sq = math.acos(1 / math.sqrt(r))       # 代码现状: sec²x = r
        x_lin = math.acos(1 / r)                 # 线性: sec x = r
        print(f"  ratio={r}: 平方版 x={math.degrees(x_sq):7.3f}° tan={math.tan(x_sq):8.4f} | "
              f"线性版 x={math.degrees(x_lin):7.3f}° tan={math.tan(x_lin):8.4f} | "
              f"螺旋√(r²-1)={math.sqrt(r*r-1):8.4f}")


if __name__ == "__main__":
    main()
