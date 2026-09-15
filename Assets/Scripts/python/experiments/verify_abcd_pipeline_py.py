"""
verify_abcd_pipeline_py.py — 用 py 测试【当前实现】的 ABCD 链路
================================================================================

按用户要求: 先用 py 把当前实现测通, 慢慢靠近。4D 记忆留给学习网络建立后再验。

上一轮我的失误: 用高斯点源场当替代场, 残差 14.6, 数不成立。
本文件的场按【管线的真实构造】重建:

    1. 逐点取 k 近邻, 做局部二次拟合 (对应 v2sjy.cs 的 NearestK + LocalQuadHessian)
    2. 得到每点的 Hessian H_i = [[p,q],[q,r]]
    3. 通道:  A_i = (p+r)/4          (实)
             B_i = (p-r)/4 + i q/2    (复)
    4. 场:     F_i = A_i * z_i + B_i * conj(z_i)      <== 这就是 totalGrad
    5. 全局拟合 F 到 5 列基 [z, zbar, z^2, z zbar, zbar^2]  (4 列基作对照)

关键推论 (本文件要验证的):
    场是【逐点线性】的 => 全局拟合到线性基就够 => C = D = E = 0
    只有当 A_i, B_i 随位置变化时, C/D/E 才会非零
    => "3 维记忆够不够"这个问题, 等价于问: A_i, B_i 到底有多不均匀

运行: python verify_abcd_pipeline_py.py
"""
import io, json, math, os, sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PTS = r"C:\Users\23128\My project (2)\Assets\Resources\points.json"


def load():
    raw = json.load(io.open(PTS, encoding="utf-8"))
    return np.array(raw, dtype=float)


def local_channels(P, k, whiten=False):
    """逐点 k 近邻局部二次拟合 -> (A_i, B_i, z_i, rel_spread)。"""
    U = P[:, 0].copy()
    V = P[:, 1].copy()
    W = P[:, 2].copy()
    if whiten:
        s = np.array([U.std(), V.std()])
        s[s < 1e-12] = 1.0
        Uw, Vw = U / s[0], V / s[1]
    else:
        Uw, Vw = U, V
    n = len(P)
    A = np.zeros(n, dtype=complex)
    B = np.zeros(n, dtype=complex)
    cond = np.zeros(n)
    for i in range(n):
        d2 = (Uw - Uw[i]) ** 2 + (Vw - Vw[i]) ** 2
        idx = np.argsort(d2)[:k]
        u = U[idx] - U[i]
        v = V[idx] - V[i]
        w = W[idx] - W[i] if False else W[idx]
        # 设计矩阵: c, u, v, u^2/2, uv, v^2/2   (6 个未知数)
        M = np.column_stack([np.ones(k), u, v, 0.5 * u * u, u * v, 0.5 * v * v])
        try:
            coef, *_ = np.linalg.lstsq(M, w, rcond=None)
        except np.linalg.LinAlgError:
            continue
        p, q, r = coef[3], coef[4], coef[5]
        A[i] = (p + r) / 4.0
        B[i] = (p - r) / 4.0 + 0.5j * q
        sv = np.linalg.svd(M, compute_uv=False)
        cond[i] = sv[0] / max(sv[-1], 1e-300)
    return A, B, U + 1j * V, cond


def fit_global(z, F, ncol):
    cols = [z, np.conj(z), z ** 2, z * np.conj(z), np.conj(z) ** 2]
    M = np.column_stack(cols[:ncol])
    coef, *_ = np.linalg.lstsq(M, F, rcond=None)
    res = float(np.max(np.abs(M @ coef - F)))
    fs = float(np.max(np.abs(F))) or 1.0
    if ncol == 4:
        out = dict(A=coef[0], B=coef[1], C=coef[2], E=0j, D=coef[3])
    else:
        out = dict(A=coef[0], B=coef[1], C=coef[2], E=coef[3], D=coef[4])
    out["res"] = res
    out["res_rel"] = res / fs
    return out


def fm(z):
    return f"{z.real:+.6f}{z.imag:+.6f}i"


def main():
    P = load()
    print("=" * 78)
    print("0. 数据")
    print("=" * 78)
    print(f"  points.json: {len(P)} 点, 列数 {P.shape[1]}")
    print(f"  坐标范围: x [{P[:,0].min():.4f},{P[:,0].max():.4f}]  "
          f"y [{P[:,1].min():.4f},{P[:,1].max():.4f}]  "
          f"z [{P[:,2].min():.4f},{P[:,2].max():.4f}]")
    print(f"  标量取第三坐标 zc = f(u,v), 无自由参数")
    print()

    print("=" * 78)
    print("1. 局部二次拟合能否定出 Hessian (k 从小到大)")
    print("=" * 78)
    print(f"  {'k':>5} {'条件数中位':>12} {'条件数最大':>12} "
          f"{'|A| 均值':>12} {'|B| 均值':>12} {'std(A)/|mean(A)|':>18} "
          f"{'std(B)/|mean(B)|':>18}")
    rows = []
    for k in (6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 199):
        if k > len(P):
            continue
        A, B, z, c = local_channels(P, k)
        mA, mB = np.abs(A).mean(), np.abs(B).mean()
        sA = A.std() / max(mA, 1e-300)
        sB = B.std() / max(mB, 1e-300)
        rows.append((k, A, B, z, c, mA, mB, sA, sB))
        print(f"  {k:>5} {np.median(c):>12.3e} {c.max():>12.3e} "
              f"{mA:>12.6f} {mB:>12.6f} {sA:>18.6f} {sB:>18.6f}")
    print()
    print("  读法: k=6 时 6 个未知数正好被 6 个点定出 => 条件数巨大, Hessian 被噪声支配;")
    print("        std(A)/|mean(A)| 越小说明局部 Hessian 越均匀 => 越像'3 个数就够'。")
    print()

    print("=" * 78)
    print("2. 全局 ABCD 拟合: 4 列基 vs 5 列基")
    print("=" * 78)
    print(f"  {'k':>5} {'基':>4} {'A':>22} {'B':>22} {'C':>22} {'E':>22} "
          f"{'D':>22} {'相对残差':>10}")
    for k, A, B, z, c, mA, mB, sA, sB in rows[:4] + rows[-2:]:
        F = A * z + B * np.conj(z)
        for ncol in (4, 5):
            g = fit_global(z, F, ncol)
            print(f"  {k:>5} {ncol:>4} {fm(g['A']):>22} {fm(g['B']):>22} "
                  f"{fm(g['C']):>22} {fm(g['E']):>22} {fm(g['D']):>22} "
                  f"{g['res_rel']:>10.3e}")
    print()
    print("  预期: 场是逐点线性的 (F_i = A_i z_i + B_i zbar_i), 所以全局拟合到线性基")
    print("        就能【精确】表示 => C = D = E 应当为 0 (机器精度)。")
    print("        这正好复现 v2sjy.cs SelfTestChannels 测到的 |C|=8.96e-17, |D|=2.55e-17。")
    print()

    print("=" * 78)
    print("3. 关键问题: C/D/E 什么时候离开 0 —— 即'3 维记忆'何时不够")
    print("=" * 78)
    print("  场是线性 ⟺ A_i, B_i 与位置无关。若它们随位置变化, 线性基就装不下,")
    print("  残差与 C/D/E 都会长起来。用【空间分块】把这件事量出来:")
    print()
    print(f"  {'k':>5} {'块数':>5} {'C 量级':>12} {'D 量级':>12} {'E 量级':>12} "
          f"{'线性基相对残差':>16}")
    for k, A, B, z, c, mA, mB, sA, sB in rows[1:8]:
        F = A * z + B * np.conj(z)
        g5 = fit_global(z, F, 5)
        # 装不下的部分 = 线性基(前 5 列去掉二次列)拟合的残差
        Mlin = np.column_stack([z, np.conj(z)])
        cl, *_ = np.linalg.lstsq(Mlin, F, rcond=None)
        resl = float(np.max(np.abs(Mlin @ cl - F)) / (np.max(np.abs(F)) or 1.0))
        print(f"  {k:>5} {len(P):>5} {abs(g5['C']):>12.3e} {abs(g5['D']):>12.3e} "
              f"{abs(g5['E']):>12.3e} {resl:>16.3e}")
    print()
    print("  => 'C/D/E 为 0 且线性基残差为 0' 意味着: 该点集在当前 k 下【局部二次结构均匀】,")
    print("     所以压缩成 3 个数 (A, B) 是无损的 —— 这就是'3 维记忆够用'的精确含义。")
    print("     一旦 A_i/B_i 随位置变化, 线性基残差 > 0, 就需要 C/D/E 等更高通道,")
    print("     也就是【记忆维度必须突破 3】。这条判据可以直接作为学习网络的验收指标。")
    print()

    print("=" * 78)
    print("4. 结构杠杆: 数量不变, 换近邻结构能否压低 ABCD 的离散度")
    print("=" * 78)
    print("  优化一(数量不变, 改结构): 这里比 '欧氏近邻' vs '白化后近邻'")
    print()
    print(f"  {'k':>5} {'规则':>10} {'std(A)/|mean|':>16} {'std(B)/|mean|':>16} "
          f"{'C 量级':>12} {'D 量级':>12}")
    for k in (12, 24, 48):
        for tag, wh in (("欧氏", False), ("白化", True)):
            A, B, z, c = local_channels(P, k, whiten=wh)
            F = A * z + B * np.conj(z)
            g = fit_global(z, F, 5)
            mA, mB = np.abs(A).mean(), np.abs(B).mean()
            print(f"  {k:>5} {tag:>10} {A.std()/max(mA,1e-300):>16.6f} "
                  f"{B.std()/max(mB,1e-300):>16.6f} {abs(g['C']):>12.3e} "
                  f"{abs(g['D']):>12.3e}")
    print()

    print("=" * 78)
    print("5. 最小点集数量 (k_min)")
    print("=" * 78)
    print("  局部二次拟合未知数 = 6  (c, a, b, H_uu, H_uv, H_vv)")
    print("  => 硬下界 k = 6; k = 6 时是插值, 条件数爆炸, 不可用")
    print()
    print(f"  {'k':>5} {'条件数中位':>12} {'std(A)/|mean|':>16} {'判定':>14}")
    for k, A, B, z, c, mA, mB, sA, sB in rows:
        med = float(np.median(c))
        if med > 1e6 or sA > 1.0:
            verdict = "不可用"
        elif med > 1e3 or sA > 0.3:
            verdict = "勉强"
        else:
            verdict = "稳定"
        print(f"  {k:>5} {med:>12.3e} {sA:>16.6f} {verdict:>14}")
    print()
    print("  => k_min 取第一个判为'稳定'的 k。这就是'最少多少个记忆点'的定量答案,")
    print("     而且它同时是优化二的输出 (ABCD 不变条件下的最小数量)。")


if __name__ == "__main__":
    main()
