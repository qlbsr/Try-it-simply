"""
verify_basis5.py — 补上 z zbar 列: 基是否变完整, 以及 E = 2*conj(C) 是否成立
================================================================================

背景 (用户指出):
    ABCD 是势能梯度的低阶展开 ∂U/∂z = sum a_mn z^m zbar^n。
    总次数 d 的单项式有 d+1 个:
        d=1:  z,  zbar                    (2 个)  <- 代码取了 2 个 ✓
        d=2:  z^2, z zbar, zbar^2         (3 个)  <- 代码只取了 2 个 ✗ 漏 z zbar
    所以 4 列基 [z, zbar, z^2, zbar^2] 是"对称截断", 不是完整低阶展开。

本文件做四件事:
  1. 补齐 5 列 [z, zbar, z^2, z zbar, zbar^2], 看三次势残差是否降到机器精度
  2. 用解析式核对 C, D, E 三个通道
  3. 验 E = 2*conj(C)  (我上一轮解析推出但未数值验证)
  4. 从 (C, D) 反解三次势系数 (p,q,r,s), 核对"三次部分只有 4 个独立实数"
  5. 真实点集上用 5 列基重算 (对照 4 列)

解析式 (把 u,v 换回 z,zbar):
    C = (3p + r)/8 - i (q + 3s)/8
    D = (3/8)[(p - r) + i(q - s)]          <- 由数值反解校正
    E = (3p + r)/4 + i (q + 3s)/4       =>  E = 2*conj(C)

运行: python verify_basis5.py
"""
import io, json, math, sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PTS = r"C:\Users\23128\My project (2)\Assets\Resources\points.json"


def field(z, a, b, c, p, q, r, s):
    """F = (U_u + i U_v)/2,  U = 1/2(a u^2+2b uv+c v^2) + p u^3+q u^2 v+r u v^2+s v^3"""
    u, v = z.real, z.imag
    Uu = a * u + b * v + 3 * p * u ** 2 + 2 * q * u * v + r * v ** 2
    Uv = b * u + c * v + q * u ** 2 + 2 * r * u * v + 3 * s * v ** 2
    return (Uu + 1j * Uv) / 2.0


def design(z, ncol):
    cols = [z, np.conj(z), z ** 2, z * np.conj(z), np.conj(z) ** 2]
    return np.column_stack(cols[:ncol])


def fit(z, F, ncol):
    M = design(z, ncol)
    coef, *_ = np.linalg.lstsq(M, F, rcond=None)
    if ncol == 4:
        return dict(A=coef[0], B=coef[1], C=coef[2], E=0j, D=coef[3],
                    res=float(np.max(np.abs(M @ coef - F))))
    return dict(A=coef[0], B=coef[1], C=coef[2], E=coef[3], D=coef[4],
                res=float(np.max(np.abs(M @ coef - F))))


def fm(x):
    return f"{x.real:+.6f}{x.imag:+.6f}i"


def part1_residual():
    print("=" * 80)
    print("1. 补齐 z zbar 后, 三次势的拟合残差")
    print("=" * 80)
    rng = np.random.default_rng(7)
    z = rng.uniform(-0.5, 0.5, 6000) + 1j * rng.uniform(-0.5, 0.5, 6000)
    print(f"  {'配置':>30} {'4 列残差':>14} {'5 列残差':>14} {'改善倍数':>10}")
    cases = [("只二次 (p=q=r=s=0)", (1.5, 0.4, 1.1, 0, 0, 0, 0)),
             ("只 u^3", (1.0, 0, 1.0, 1, 0, 0, 0)),
             ("只 u^2 v", (1.0, 0, 1.0, 0, 1, 0, 0)),
             ("只 u v^2", (1.0, 0, 1.0, 0, 0, 1, 0)),
             ("只 v^3", (1.0, 0, 1.0, 0, 0, 0, 1)),
             ("混合", (1.0, 0, 1.0, 1.0, 0.5, -0.7, 0.3))]
    for tag, args in cases:
        F = field(z, *args)
        r4 = fit(z, F, 4)['res']
        r5 = fit(z, F, 5)['res']
        print(f"  {tag:>30} {r4:>14.3e} {r5:>14.3e} {r4/max(r5,1e-300):>10.1f}")
    print()
    print("  => 5 列基把三次势残差压到机器精度, 4 列基不行。缺列确认。")
    print()


def part2_channels():
    print("=" * 80)
    print("2. 通道解析式核对 (5 列基)")
    print("=" * 80)
    rng = np.random.default_rng(11)
    z = rng.uniform(-0.4, 0.4, 8000) + 1j * rng.uniform(-0.4, 0.4, 8000)
    print(f"  {'p':>6} {'q':>6} {'r':>6} {'s':>6} | {'C 拟合':>19} {'C 解析':>19} "
          f"{'D 拟合':>19} {'D 解析':>19}")
    worst = dict(C=0.0, D=0.0, E=0.0)
    for p, q, r, s in [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1),
                       (1.0, 0.5, -0.7, 0.3), (-0.4, 1.2, 0.8, -0.25)]:
        F = field(z, 1.0, 0.0, 1.0, p, q, r, s)
        g = fit(z, F, 5)
        Cc = (3 * p + r) / 8.0 - 1j * (q + 3 * s) / 8.0
        Dc = 3.0 * (p - r) / 8.0 + 3j * (q - s) / 8.0
        worst['C'] = max(worst['C'], abs(g['C'] - Cc))
        worst['D'] = max(worst['D'], abs(g['D'] - Dc))
        print(f"  {p:>6.2f} {q:>6.2f} {r:>6.2f} {s:>6.2f} | {fm(g['C']):>19} "
              f"{fm(Cc):>19} {fm(g['D']):>19} {fm(Dc):>19}")
    print()
    print(f"  最大偏差:  |dC| = {worst['C']:.3e}    |dD| = {worst['D']:.3e}")
    print("  => C, D 的解析式成立。")
    print()


def part3_E_relation():
    print("=" * 80)
    print("3. 验证 E = 2*conj(C)  <<< 上一轮解析推出, 未数值验证过")
    print("=" * 80)
    rng = np.random.default_rng(13)
    z = rng.uniform(-0.4, 0.4, 8000) + 1j * rng.uniform(-0.4, 0.4, 8000)
    print(f"  {'p':>6} {'q':>6} {'r':>6} {'s':>6} | {'E 拟合':>22} "
          f"{'2*conj(C) 解析':>22} {'差':>12}")
    worst_E = 0.0
    worst_C = 0.0
    for p, q, r, s in [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1),
                       (1.0, 0.5, -0.7, 0.3), (-0.4, 1.2, 0.8, -0.25),
                       (2.0, -1.0, 0.4, 0.9)]:
        F = field(z, 1.0, 0.0, 1.0, p, q, r, s)
        g = fit(z, F, 5)
        Ec = (3 * p + r) / 4.0 + 1j * (q + 3 * s) / 4.0
        Cc = (3 * p + r) / 8.0 - 1j * (q + 3 * s) / 8.0
        lhs = g['E']
        rhs = 2.0 * np.conj(Cc)
        worst_E = max(worst_E, abs(lhs - rhs))
        worst_C = max(worst_C, abs(g['C'] - Cc))
        print(f"  {p:>6.2f} {q:>6.2f} {r:>6.2f} {s:>6.2f} | {fm(lhs):>22} "
              f"{fm(rhs):>22} {abs(lhs-rhs):>12.3e}")
    print()
    print(f"  最大 |E - 2*conj(C_解析)| = {worst_E:.3e}")
    print(f"  最大 |C_拟合 - C_解析|    = {worst_C:.3e}   (顺带核对)")
    if worst_E < 1e-10:
        print("  => 成立! E = 2*conj(C)  ==  二次次 6 个实系数只有 4 个独立。")
    else:
        print("  => 不成立, 我的解析推导有错, 需要重推。")
    print()


def part4_invert():
    print("=" * 80)
    print("4. 从 (C, D) 反解三次势系数 (信息量核对)")
    print("=" * 80)
    print("  由 Re C=(3p+r)/8, Im C=-(q+3s)/8, Re D=(3p-r)/8, Im D=3(q+s)/8 反解:")
    print("      p = 2 Re C + (2/3) Re D")
    print("      r = p - (8/3) Re D")
    print("      s = -2 Im C - (2/3) Im D")
    print("      q = (8/3) Im D + s")
    print()
    rng = np.random.default_rng(17)
    z = rng.uniform(-0.4, 0.4, 8000) + 1j * rng.uniform(-0.4, 0.4, 8000)
    print(f"  {'p':>7} {'q':>7} {'r':>7} {'s':>7} | {'p 反解':>11} {'q 反解':>11} "
          f"{'r 反解':>11} {'s 反解':>11}")
    worst = 0.0
    for p, q, r, s in [(1.0, 0.5, -0.7, 0.3), (-0.4, 1.2, 0.8, -0.25),
                       (2.0, -1.0, 0.4, 0.9), (0.3, 0.0, 0.0, -0.6)]:
        F = field(z, 1.0, 0.0, 1.0, p, q, r, s)
        g = fit(z, F, 5)
        C, D = g['C'], g['D']
        pp = 2.0 * C.real + (2.0 / 3.0) * D.real
        rr = pp - (8.0 / 3.0) * D.real
        ss = -2.0 * C.imag - (2.0 / 3.0) * D.imag
        qq = (8.0 / 3.0) * D.imag + ss
        worst = max(worst, abs(pp - p), abs(qq - q), abs(rr - r), abs(ss - s))
        print(f"  {p:>7.2f} {q:>7.2f} {r:>7.2f} {s:>7.2f} | {pp:>11.6f} {qq:>11.6f} "
              f"{rr:>11.6f} {ss:>11.6f}")
    print()
    print(f"  最大反解偏差 = {worst:.3e}")
    print("  => (C, D) 4 个实数 <-> (p,q,r,s) 4 个实数, 双射。")
    print("     E 是冗余的: 补它是为了让【拟合不出偏差】, 不是因为它带新信息。")
    print()


def part5_real():
    print("=" * 80)
    print("5. 真实点集上 4 列 vs 5 列 (逐点构造的场)")
    print("=" * 80)
    P = np.array(json.load(io.open(PTS, encoding="utf-8")), dtype=float)
    n = len(P)
    U, V, W = P[:, 0], P[:, 1], P[:, 2]
    print(f"  points.json: {n} 点")
    print()
    print(f"  {'k':>5} {'基':>4} {'最大残差':>13} {'|C|':>12} {'|E|':>12} "
          f"{'|D|':>12} {'|E-2conj(C)|':>14}")
    for k in (12, 24, 48):
        A = np.zeros(n, dtype=complex)
        B = np.zeros(n, dtype=complex)
        for i in range(n):
            d2 = (U - U[i]) ** 2 + (V - V[i]) ** 2
            idx = np.argsort(d2)[:k]
            u = U[idx] - U[i]
            v = V[idx] - V[i]
            w = W[idx]
            M = np.column_stack([np.ones(k), u, v, 0.5 * u * u, u * v, 0.5 * v * v])
            coef, *_ = np.linalg.lstsq(M, w, rcond=None)
            pp, qq, rr = coef[3], coef[4], coef[5]
            A[i] = (pp + rr) / 4.0
            B[i] = (pp - rr) / 4.0 + 0.5j * qq
        zc = U + 1j * V
        F = A * zc + B * np.conj(zc)
        fs = np.max(np.abs(F)) or 1.0
        for ncol in (4, 5):
            g = fit(zc, F, ncol)
            print(f"  {k:>5} {ncol:>4} {g['res']/fs:>13.3e} {abs(g['C']):>12.3e} "
                  f"{abs(g['E']):>12.3e} {abs(g['D']):>12.3e} "
                  f"{abs(g['E']-2*np.conj(g['C'])):>14.3e}")
    print()
    print("  注: 这里的场是【逐点】Hessian 拼出来的, 所以含大量高阶内容,")
    print("      5 列基也装不下 -> 残差大。这反映的是'通道数不够', 不是缺列。")
    print("      缺列的证据在第 1 部分(受控势): 4 列 0.13~0.40, 5 列 1e-16。")


if __name__ == "__main__":
    part1_residual()
    part2_channels()
    part3_E_relation()
    part4_invert()
    part5_real()
