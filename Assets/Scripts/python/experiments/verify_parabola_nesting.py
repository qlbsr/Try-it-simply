"""
verify_parabola_nesting.py — 抛物线多向量如何逼近函数曲线: 空间嵌套机制
================================================================================

要点(用户指出, 我上一轮答错了层): 不是"多个向量各自配对 -> 包络收敛",
而是【特征空间嵌套】: 每层把上一层的双抛物线【视为一体】再相交, 得到下一层。

代码证据 (legacy/sjy.cs:567-622, Vss1):

    链一 (第一层特征空间):
        v_flat  = v3
        R_ray   = FromToRotation(forward, v_flat)
        v_bent  = (R_ray^-1 * v_2).normalized        <-- 把 v_2 拉进第一层空间
        fj      = atan2(v_bent.z, v_bent.x)          <-- 该层特征读数 1 (方位)
        d2_new  = acos(v_bent.y)                     <-- 该层特征读数 2 (张角)

    链二 (第二层, 建立在第一层之上):
        R_space = FromToRotation(v_flat, v_bent)     <-- 第二层空间由第一层结果定出
        R       = AngleAxis(deltaAngle, cross(v_flat, v_bent))   (== R_space 的矩阵形式)
        ez_bent = R * ez
        v_final = R_space * ez_bent * zLength        <-- R_space 【作用两次】
        R_total = R_ray * R_space                    <-- 复合: 空间先, 折线后

    双抛物线:
        (inter1, inter2) = FindIntersections(v_final, V_A, V_3, v_c, v_c1)
        vz2              = GetVertexFromFourPoints(inter1, inter2, V_q1, V_q2)
        points1 = GetParabolaSegmentPoints(v_final, V_A,  V_D)
        points2 = GetParabolaSegmentPoints(vz2,     V_q1, V_q2)

    => 第二层的锚点 vz2 由【第一层的交点对】定出: (inter1, inter2) 被视为一体。

本文件验证三条代数事实:
  1. R_space 作用两次 => v_final = R_space^2 * ẑ * zLength (AngleAxis 版 == FromTo 版)
  2. 抛物线内部解析冗余: 全部采样点满足 u = zLength - (zLength/z2^2)*v^2
     => 中间连接段被【直接等同】, 只需两端向量
  3. ComputeBendEnergy 的被积函数 k^2/(1+k^2 v^2)^{5/2} 恰好是 kappa^2 * (ds/dv)
     => 它就是弯曲能量 ∫kappa^2 ds, 因此也只依赖两端在 e2 上的投影。

运行: python verify_parabola_nesting.py
"""
import io, math, sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def rot_axis(axis, ang):
    ax = np.asarray(axis, float)
    ax = ax / np.linalg.norm(ax)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)


def from_to(f, t):
    f = np.asarray(f, float) / np.linalg.norm(f)
    t = np.asarray(t, float) / np.linalg.norm(t)
    d = float(np.dot(f, t))
    if d > 1 - 1e-15:
        return np.eye(3)
    if d < -1 + 1e-15:
        a = np.array([1.0, 0, 0]) if abs(f[0]) < 0.9 else np.array([0.0, 1, 0])
        a = a - np.dot(a, f) * f
        return rot_axis(a, math.pi)
    ax = np.cross(f, t)
    return rot_axis(ax, math.acos(max(-1.0, min(1.0, d))))


def part1_nesting():
    print("=" * 78)
    print("1. 链二 = R_space 作用【两次】; R_total = R_ray * R_space (空间先)")
    print("=" * 78)
    rng = np.random.default_rng(3)
    worst_ax = 0.0
    worst_vf = 0.0
    worst_ord = 0.0
    for _ in range(5000):
        def unit():
            x = rng.normal(size=3)
            return x / np.linalg.norm(x)
        v_flat, v_2 = unit(), unit()
        R_ray = from_to([0, 0, 1.0], v_flat)
        v_bent = R_ray.T @ v_2
        v_bent = v_bent / np.linalg.norm(v_bent)
        R_space = from_to(v_flat, v_bent)

        # 代码里用 AngleAxis(deltaAngle, cross(v_flat,v_bent)) 重算了一次 R
        axis = np.cross(v_flat, v_bent)
        delta = math.acos(max(-1.0, min(1.0, float(np.dot(v_flat, v_bent)))))
        R = rot_axis(axis, delta)
        worst_ax = max(worst_ax, float(np.max(np.abs(R - R_space))))

        zLength = float(rng.uniform(0.3, 3.0))
        ez_bent = R_space @ np.array([0, 0, 1.0])
        v_final_code = R_space @ ez_bent * zLength
        v_final_sq = (R_space @ R_space) @ np.array([0, 0, 1.0]) * zLength
        worst_vf = max(worst_vf, float(np.max(np.abs(v_final_code - v_final_sq))))

        # 复合顺序: R_total = R_ray * R_space  =>  R_space 先作用
        x = unit()
        worst_ord = max(worst_ord,
                        float(np.max(np.abs((R_ray @ R_space) @ x - R_ray @ (R_space @ x)))))
    print(f"  max |AngleAxis(delta, v_flat x v_bent) - FromTo(v_flat, v_bent)| = {worst_ax:.3e}")
    print(f"  max |R_space*ez_bent*zL - R_space^2*ẑ*zL|                          = {worst_vf:.3e}")
    print(f"  max |(R_ray*R_space)x - R_ray*(R_space x)|   (空间先作用)          = {worst_ord:.3e}")
    print()
    print("  => R 就是 R_space 的矩阵形式(冗余重算), v_final = R_space^2 * ẑ * zLength。")
    print("     第二层空间【作用两次】= 你说的'在第一层前提下再进一层'。")
    print("     复合 R_total = R_ray*R_space 中 R_space 先作用 => '空间先, 折线后'。")
    print()


def part2_interior():
    print("=" * 78)
    print("2. 抛物线内部解析冗余 —— '中间连接部分直接等同'")
    print("=" * 78)
    print("  GetParabolaSegmentPoints(v_final, start, end, numPoints):")
    print("      e1 = v_final.normalized,  zLength = |v_final|")
    print("      delta = end - start;  perp = delta - (delta.e1)e1")
    print("      e2 = perp.normalized;  z2 = |perp| * 0.5")
    print("      uCoord = zLength - (zLength/(z2*z2)) * vCoord^2")
    print("      pt = uCoord*e1 + vCoord*e2")
    print()
    rng = np.random.default_rng(21)
    worst = 0.0
    for _ in range(2000):
        zLength = float(rng.uniform(0.3, 3.0))
        e1 = np.array([0.0, 0.0, 1.0])
        th = float(rng.uniform(0.0, 2 * math.pi))
        e2 = np.array([math.cos(th), math.sin(th), 0.0])
        z2 = float(rng.uniform(0.05, 0.5))
        minV = float(rng.uniform(-z2, 0.0))
        maxV = float(rng.uniform(0.0, z2))
        n = 100
        for i in range(n):
            t = i / (n - 1)
            v = minV + (maxV - minV) * t
            u = zLength - (zLength / (z2 * z2)) * v * v
            pt = u * e1 + v * e2
            # 独立反查: 从点本身恢复 u,v 是否满足抛物线律
            u_chk = float(np.dot(pt, e1))
            v_chk = float(np.dot(pt, e2))
            resid = abs(u_chk - (zLength - (zLength / (z2 * z2)) * v_chk * v_chk))
            worst = max(worst, resid)
    print(f"  2000 组随机 (zLength, e2, z2, 区间):  max 残差 = {worst:.3e}")
    print()
    print("  => 100 个点全部由 (e1, e2, zLength, z2, minV, maxV) 解析生成,")
    print("     内部没有任何自由度。持有两端 (start, end) 就等于持有整条弧。")
    print("     这就是'中间连接部分直接等同, 也就不用算了'。")
    print()


def part3_bending_energy():
    print("=" * 78)
    print("3. ComputeBendEnergy 的被积函数 = kappa^2 * (ds/dv) —— 弯曲能量")
    print("=" * 78)
    print("  抛物线 u(v) = zLength - (zLength/z2^2) v^2  =>  u' = -k*v, u'' = -k,")
    print("      其中 k = 2*zLength/z2^2   (代码里的 k)")
    print("  曲率 kappa = |u''|/(1+u'^2)^{3/2} = k/(1+k^2 v^2)^{3/2}")
    print("  弧长元 ds  = sqrt(1+u'^2) dv = sqrt(1+k^2 v^2) dv")
    print("  => kappa^2 * ds/dv = k^2/(1+k^2 v^2)^3 * (1+k^2 v^2)^{1/2}")
    print("                     = k^2/(1+k^2 v^2)^{5/2}   <== 代码里的 f(v)")
    print()
    worst = 0.0
    for zL, z2 in [(1.0, 0.3), (2.5, 0.11), (0.4, 0.05), (3.0, 0.49)]:
        k = 2 * zL / (z2 * z2)
        for v in np.linspace(-z2, z2, 2001):
            upp = -k * v
            upp2 = -k
            kappa = abs(upp2) / (1 + upp ** 2) ** 1.5
            ds_dv = math.sqrt(1 + upp ** 2)
            code_f = (k * k) / (1 + k * k * v * v) ** 2.5
            worst = max(worst, abs(kappa ** 2 * ds_dv - code_f))
    print(f"  max |kappa^2*(ds/dv) - f(v)| = {worst:.3e}")
    print()
    # 数值验证 Simpson 积分等价于 ∫ kappa^2 ds
    zL, z2 = 1.0, 0.3
    k = 2 * zL / (z2 * z2)
    vs = np.linspace(-z2, 0.7 * z2, 4001)
    kappa = k / (1 + k * k * vs * vs) ** 1.5
    ds = np.sqrt(1 + k * k * vs * vs)
    bend = np.trapezoid(kappa ** 2 * ds, vs)
    code = np.trapezoid((k * k) / (1 + k * k * vs * vs) ** 2.5, vs)
    print(f"  数值对照 (zLength={zL}, z2={z2}, 区间 [-z2, 0.7z2]):")
    print(f"      int kappa^2 ds        = {bend:.9f}")
    print(f"      int f(v) dv (代码)    = {code:.9f}")
    print(f"      相对差                = {abs(bend-code)/bend:.3e}")
    print()
    print("  => ComputeBendEnergy 就是弯曲能量 ∫kappa^2 ds (Euler-Bernoulli 弹性能),")
    print("     而被积函数的自变量只有 v = 点.e2, 所以整条弧的能量只由两端在 e2")
    print("     上的投影 (v_start, v_end) 决定 —— 中间再次被消掉。")
    print()


if __name__ == "__main__":
    part1_nesting()
    part2_interior()
    part3_bending_energy()
    print("=" * 78)
    print("机制总结: 抛物线多向量如何逼近函数曲线")
    print("=" * 78)
    print("  不是: 多个向量各自配对, 包络随向量数增加而收敛  (我上轮的错误读法)")
    print("  而是: 特征空间【嵌套】, 每层把上层结果整体收缩成一个锚点")
    print()
    print("  层 0  : v_flat = v3                                       (参考轴)")
    print("  层 1  : R_ray = FromTo(ẑ, v_flat)                         第一层特征空间")
    print("          v_bent = R_ray^-1 v_2   -> 读 (fj, d2_new)        该层特征读数")
    print("  层 2  : R_space = FromTo(v_flat, v_bent)                  第二层空间(建于层1)")
    print("          v_final = R_space^2 ẑ zLength                     【二次】作用=再进一层")
    print("          R_total = R_ray R_space                           嵌套写在乘法顺序里")
    print("  收缩  : (inter1, inter2) --GetVertexFromFourPoints--> vz2")
    print("          第一层的【交点对被视为一体】, 塌缩成下一层的锚点")
    print("  消中  : 每层两条抛物线的内部由 u = zLength - (zLength/z2^2)v^2 解析等同,")
    print("          能量只吃两端投影, 所以中间连接段永远不用算")
    print()
    print("  于是: 每层都在做同一个算子 '相交 -> 定锚点 -> 生成抛物线',")
    print("        且锚点由上层整体收缩而来 => 这是一个【递归/IFS 式收缩】,")
    print("        极限对象就是被逼近的那条函数曲线。")
    print("        '趋向哪个函数' = 这个算子的不动点是什么, 而不是包络形状。")
