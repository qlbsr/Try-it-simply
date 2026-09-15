"""
verify_abcd_meaning.py — 逐通道扫描: A / B / C / D 到底各代表什么
================================================================================

代码约定 (v2sjy.cs FitChannels):
    拟合 F = A*z + B*zbar + C*z^2 + D*zbar^2      基 = [z, zbar, z^2, zbar^2]
    z = u + i v

本文件用【受控势场】反推每个通道的身份, 而不是靠读注释。

受控势场:
    U(u,v) = 1/2(a u^2 + 2b uv + c v^2)  +  p u^3 + q u^2 v + r u v^2 + s v^3
    复数场取 F = (U_u + i U_v)/2   (这个 1/2 让二次部分正好等于代码里的 A=alpha)

运行: python verify_abcd_meaning.py
"""
import io, math, sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def field_and_basis(a, b, c, p, q, r, s, Z):
    """Z: (n,) complex 采样点。返回 (F, 设计矩阵 [z, zbar, z^2, zbar^2])。"""
    u, v = Z.real, Z.imag
    Uu = a * u + b * v + 3 * p * u ** 2 + 2 * q * u * v + r * v ** 2
    Uv = b * u + c * v + q * u ** 2 + 2 * r * u * v + 3 * s * v ** 2
    F = (Uu + 1j * Uv) / 2.0
    M = np.column_stack([Z, np.conj(Z), Z ** 2, np.conj(Z) ** 2])
    return F, M


def fit(a, b, c, p=0.0, q=0.0, r=0.0, s=0.0, n=4000, seed=1):
    rng = np.random.default_rng(seed)
    Z = rng.uniform(-0.5, 0.5, n) + 1j * rng.uniform(-0.5, 0.5, n)
    F, M = field_and_basis(a, b, c, p, q, r, s, Z)
    coef, res, rank, sv = np.linalg.lstsq(M, F, rcond=None)
    resid = float(np.max(np.abs(M @ coef - F)))
    return dict(A=coef[0], B=coef[1], C=coef[2], D=coef[3],
                resid=resid, rank=rank, sv=sv)


def part1_hessian():
    print("=" * 78)
    print("1. A 与 B 的身份: 它们把对称 2x2 Hessian 完整重参数化")
    print("=" * 78)
    print("  H = [[a, b], [b, c]]   (实对称, 3 个自由度)")
    print("  代码: alpha = (H_uu + H_vv)/4 = (a+c)/4   ->  A = alpha")
    print("        beta  = (H_uu - H_vv + 2i H_uv)/4  ->  B = beta")
    print()
    print(f"  {'a':>7} {'b':>7} {'c':>7} | {'Im A (应=0)':>12} "
          f"{'Re B':>10} {'(a-c)/4':>10} {'Im B':>10} {'b/2':>10} "
          f"{'|A|^2-|B|^2':>12} {'det/4':>10} {'拟合残差':>10}")
    for a, b, c in [(2.0, 0.0, 2.0), (3.0, 0.0, 1.0), (1.0, 0.8, 1.0),
                    (2.0, 0.0, -1.0), (0.5, 0.5, 0.5), (4.0, 1.5, -2.0)]:
        r = fit(a, b, c)
        A, B = r['A'], r['B']
        lhs = abs(A) ** 2 - abs(B) ** 2
        det4 = (a * c - b * b) / 4.0
        print(f"  {a:>7.2f} {b:>7.2f} {c:>7.2f} | {A.imag:>12.3e} "
              f"{B.real:>10.6f} {(a-c)/4:>10.6f} {B.imag:>10.6f} {b/2:>10.6f} "
              f"{lhs:>12.6f} {det4:>10.6f} {r['resid']:>10.2e}")
    print()
    print("  => Im A 恒为 0: 【A 是实数, 只有 1 个实自由度】")
    print("     Re B = (a-c)/4  各向异性(对角差)")
    print("     Im B = b/2      剪切耦合(非对角)")
    print("     |A|^2 - |B|^2 = det(H)/4")
    print()
    print("  自由度核对:  A(1 实) + B(2 实) = 3 实 = 对称 2x2 Hessian 的自由度  <== 严密对应")
    print()


def part2_classify():
    print("=" * 78)
    print("2. |A| 与 |B| 之比 = Hessian 类型 (这是 A,B 最有用的读法)")
    print("=" * 78)
    print("  |A| > |B|  <=>  det H > 0  <=>  椭圆型(正定/负定)  <=>  稳定")
    print("  |A| = |B|  <=>  det H = 0  <=>  抛物型(退化)       <=>  有平坦方向")
    print("  |A| < |B|  <=>  det H < 0  <=>  双曲型(鞍点)       <=>  不稳定")
    print()
    print(f"  {'a':>7} {'b':>7} {'c':>7} | {'|A|':>10} {'|B|':>10} "
          f"{'|A|-|B|':>12} {'det H':>10} {'类型':>8}")
    cases = [(2.0, 0.0, 2.0), (3.0, 0.0, 1.0), (2.0, 0.0, -1.0),
             (1.0, 1.0, 1.0), (1.0, 2.0, 1.0), (0.0, 0.0, 0.0)]
    for a, b, c in cases:
        r = fit(a, b, c)
        A, B = r['A'], r['B']
        det = a * c - b * b
        if abs(abs(A) - abs(B)) < 1e-9:
            kind = "抛物"
        elif abs(A) > abs(B):
            kind = "椭圆"
        else:
            kind = "双曲"
        print(f"  {a:>7.2f} {b:>7.2f} {c:>7.2f} | {abs(A):>10.6f} {abs(B):>10.6f} "
              f"{abs(A)-abs(B):>12.3e} {det:>10.6f} {kind:>8}")
    print()
    print("  => |A| = |B| 就是【抛物型】。注意: 你的结构用的正是抛物线,")
    print("     所以这个判据直接对应'结构本身处在抛物边界上'。")
    print()


def part3_cubic():
    print("=" * 78)
    print("3. C 与 D 的身份: 势能的三次部分, 以及基的缺项")
    print("=" * 78)
    print("  实三次势 p u^3 + q u^2 v + r u v^2 + s v^3 有 4 个实自由度。")
    print("  解析推导(把 u,v 换回 z,zbar)给出:")
    print("      C = (3p + r)/8 - i (q + 3s)/8")
    print("      D = (3p - r)/8 + 3i (q + s)/8")
    print()
    print(f"  {'p':>6} {'q':>6} {'r':>6} {'s':>6} | {'C 拟合':>20} {'C 解析':>20} "
          f"{'D 拟合':>20} {'D 解析':>20}")
    for p, q, r, s in [(1.0, 0, 0, 0), (0, 0, 1.0, 0), (0, 1.0, 0, 0),
                       (0, 0, 0, 1.0), (1.0, 0.5, -0.7, 0.3)]:
        rf = fit(1.0, 0.0, 1.0, p, q, r, s)
        Cc = (3 * p + r) / 8.0 - 1j * (q + 3 * s) / 8.0
        Dc = (3 * p - r) / 8.0 + 3j * (q + s) / 8.0
        def fm(z):
            return f"{z.real:+.6f}{z.imag:+.6f}i"
        print(f"  {p:>6.2f} {q:>6.2f} {r:>6.2f} {s:>6.2f} | {fm(rf['C']):>20} "
              f"{fm(Cc):>20} {fm(rf['D']):>20} {fm(Dc):>20}")
    print()
    print("  => C, D 精确等于三次势的系数组合: 4 个实自由度 = C(2 实) + D(2 实)  <== 严密对应")
    print()
    print("  【基的缺项】一般实三次势的梯度含 zzbar = |z|^2 项, 而基 [z,zbar,z^2,zbar^2]")
    print("  没有这一项。检验: 拟合残差在含 zzbar 的势下会不会上升?")
    print()
    print(f"  {'配置':>34} {'最大残差':>14} {'秩':>5}")
    tests = [("只二次 (p=q=r=s=0)", (1.5, 0.4, 1.1, 0, 0, 0, 0)),
             ("只 u^3 (p=1)", (1.0, 0, 1.0, 1, 0, 0, 0)),
             ("只 u^2 v (q=1)", (1.0, 0, 1.0, 0, 1, 0, 0)),
             ("只 u v^2 (r=1)", (1.0, 0, 1.0, 0, 0, 1, 0)),
             ("只 v^3 (s=1)", (1.0, 0, 1.0, 0, 0, 0, 1)),
             ("混合 (p,q,r,s 全非零)", (1.0, 0, 1.0, 1.0, 0.5, -0.7, 0.3))]
    for tag, args in tests:
        r = fit(*args)
        print(f"  {tag:>34} {r['resid']:>14.3e} {r['rank']:>5}")
    print()
    print("  秩恒为 4 = 基的列数, 说明基内不退化; 残差见上。")
    print()


def part4_delta():
    print("=" * 78)
    print("4. Delta = (B^2 - 4AC)/(D^2 - 4BC) 在当前区制下是否有定义")
    print("=" * 78)
    print("  我前面已验证 (v2sjy SelfTestChannels): 常 Hessian 时 |C|=8.96e-17, |D|=2.55e-17。")
    print("  也就是说当前管线的势【就是二次的】, C = D = 0。")
    print()
    print("  代入 Delta:")
    print("      B^2 - 4AC  = B^2 - 0 = B^2")
    print("      D^2 - 4BC  = 0   - 0 = 0")
    print("      Delta = B^2 / 0     <== 未定义")
    print()
    print(f"  {'C':>14} {'D':>14} {'B':>14} {'Delta':>24}")
    for C, D, B in [(0.0, 0.0, 0.5 + 0.3j), (1e-17, 2e-17, 0.5 + 0.3j),
                    (1e-6, 1e-6, 0.5 + 0.3j), (0.3, -0.2, 0.5 + 0.3j)]:
        num = B * B - 4 * (0.4) * C
        den = D * D - 4 * B * C
        d = "(未定义 0/0 或不稳)" if abs(den) < 1e-14 else f"{num/den:+.9f}"
        print(f"  {C:>14.3e} {D:>14.3e} {B:>14.6f} {d:>24}")
    print()
    print("  => Delta 要【有定义】必须先有 C 或 D 非零, 即势必须是非二次的。")
    print("     当前管线 C = D = 0 (机器精度), 所以 Delta -> tau 这条路现在是【结构性堵住】的。")
    print("     要打通它, 必须让势离开二次(抛物)区制 —— 而'离开抛物区制'正是前面")
    print("     verify_least_action_anchor.py 里测到的: 真解比抛物线能量低 54.87%。")
    print()


def part5_table():
    print("=" * 78)
    print("汇总表: 每个部分真正代表什么")
    print("=" * 78)
    rows = [
        ("A", "实 (Im A = 0)", "1 个实自由度",
         "(H_uu+H_vv)/4 = 迹/4 = 平均对角 = 各向同性刚度",
         "det H > 0 的一半判据; A 单独 = 圆形弹簧"),
        ("B", "复", "2 个实自由度",
         "Re B=(H_uu-H_vv)/4 各向异性;  Im B=H_uv/2 剪切耦合",
         "|A| vs |B| 给出 Hessian 类型: 椭圆/抛物/双曲"),
        ("A,B 合计", "—", "3 个实自由度",
         "对称 2x2 Hessian 的【完整最小重参数化】",
         "|A|^2-|B|^2 = det(H)/4"),
        ("C", "复", "2 个实自由度",
         "势能三次项的 z^2 通道: (3p+r)/8 - i(q+3s)/8",
         "衡量'离开抛物区制'的程度"),
        ("D", "复", "2 个实自由度",
         "势能三次项的 zbar^2 通道: (3p-r)/8 + 3i(q+s)/8",
         "同上, 另一半"),
        ("C,D 合计", "—", "4 个实自由度",
         "实三次势 p u^3+q u^2 v+r u v^2+s v^3 的【完整最小重参数化】",
         "C=D=0  <=>  势是纯二次  <=>  结构是纯抛物"),
    ]
    print(f"  {'通道':>9} {'类型':>14} {'自由度':>16} {'代表什么':>44}")
    for t, ty, dof, mean, _ in rows:
        print(f"  {t:>9} {ty:>14} {dof:>16} {mean:>44}")
    print()
    print("  用法:")
    for t, _, _, _, use in rows:
        print(f"    {t:>9}: {use}")
    print()
    print("  总计: A(1)+B(2)+C(2)+D(2) = 7 个实自由度 = 二次势 3 + 三次势 4  <== 恰好用完")
    print("  也就是说 ABCD 不是四个独立物理量, 而是【势场低阶展开的完备坐标系】:")
    print("      二次部分(3 自由度) -> A, B      三次部分(4 自由度) -> C, D")
    print()
    print("  缺项: 基 [z, zbar, z^2, zbar^2] 没有 |z|^2 = z zbar 这一项。")
    print("        一般实三次势的梯度含 zzbar, 所以这个基不是完备的 5 项二次基。")
    print("        若要让 C, D 干净地只承载三次信息, 基应补成")
    print("            [z, zbar, z^2, z zbar, zbar^2]")


if __name__ == "__main__":
    part1_hessian()
    part2_classify()
    part3_cubic()
    part4_delta()
    part5_table()
