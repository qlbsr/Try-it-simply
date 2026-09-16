"""
verify_vss_two_sheet.py — 两条叶的代数结构（已实测确认）
================================================================================

背景与结论（本文件就是那次实测的可复现版本）:

  Vss1 的两个关系向量:
      v3   = 圆锥轴                    （字段）
      vss  = 概率流线主方向 / 压缩残差主方向（入参 v，FitDispersion 的输出）

  两条叶（修正后）:
      V_3 = normalize( cross(v_final, vss) )
      n±  = normalize( v_final ± V_3 * z2 )

  其中  zLength = sqrt(r[2]),  z2 = 1 - zLength
        （单位数据上 Σr[i] ≈ 1, 故 zLength ∈ (0,1), z2 ∈ (0,1),
          且 zLength + z2 = sqrt(Σr[i]) = 1 —— 即「轴向分量 + 扭曲分量 = 单位长度」）

  实测确认的三条:
      ① V_3 ⊥ v_final   且   V_3 ⊥ vss                      偏差 ~5e-16
      ② 间隔 d = 2*z2/sqrt(L^2 + z2^2)  精确（L = zLength）   偏差 6.7e-16
      ③ d 与方向【完全无关】: 让 V_3 跑遍整个单位球面,
         20000 组样本的 d 标准差 1.4e-16, 与 V_3 各分量相关性 ~0

  ⟹ 意义: 两条叶的「间隔」只编码残差的【大小】(纯 r[2] 的函数);
          「朝向」V_3 才编码残差的【方向】。两者职责分离。
          两条叶严格关于 v_final 对称 ⟹ 上下旋无法由叶面区分,
          区分只能来自两段弧上的能量/曲率（κ⁻ = κ⁺）。

本文件曾有的三处错误（已在上一轮更正，这里记录以免重犯）:
  1. 造点用单位 v_final —— 应 |v_final| = zLength（代码里 v_final = R_space*(R_space*ẑ)*zLength）
     用单位向量会测出 0.574695771（对应 2z2/sqrt(1+z2^2)），从而误报「公式偏差 0.697」
  2. 测试 V_3 变化时把 vss 限制在 xz 平面且 vf = ẑ ⟹ V_3 ≡ ŷ 恒定, 实际没在变
  3. 造锥面点时半顶角用了 d2 —— 应为 theta2（theta2 才是锥半顶角）
     且锥轴是 y（th2 把 H 放在 y 分量），不是 z

运行: python verify_vss_two_sheet.py
"""
import io, math, sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

SEED = 7
N = 20000


def unit(rng, n=3):
    x = rng.normal(size=n)
    return x / np.linalg.norm(x)


def main():
    rng = np.random.default_rng(SEED)

    # ---------------- ① 正交恒等式 + 间隔闭式 ----------------
    w1 = w2 = w3 = 0.0
    cnt = 0
    for _ in range(N):
        L = float(rng.uniform(0.05, 0.99))
        z2 = 1.0 - L
        vf = unit(rng) * L                       # |v_final| = zLength  ← 与代码一致
        vss = unit(rng)
        c = np.cross(vf, vss)
        if np.linalg.norm(c) < 1e-9:
            continue
        V3 = c / np.linalg.norm(c)
        w1 = max(w1, abs(float(V3 @ vf)))
        w2 = max(w2, abs(float(V3 @ vss)))
        a, b = vf + V3 * z2, vf - V3 * z2
        a /= np.linalg.norm(a)
        b /= np.linalg.norm(b)
        pred = 2.0 * z2 / math.sqrt(L * L + z2 * z2)
        w3 = max(w3, abs(float(np.linalg.norm(a - b)) - pred))
        cnt += 1

    print("=" * 76)
    print("(1) 正交恒等式 与 间隔闭式          [已实测确认]")
    print("=" * 76)
    print(f"  样本数 = {cnt}")
    print(f"  max |V_3 · v_final|                 = {w1:.4e}")
    print(f"  max |V_3 · vss|                     = {w2:.4e}")
    print(f"  max | 间隔 - 2z2/sqrt(L^2+z2^2) |    = {w3:.4e}")
    print("  ⟹ V_3 同时垂直于两个关系向量; 间隔闭式精确成立。")
    print()

    # ---------------- ② 间隔与方向无关 ----------------
    L = 0.7
    z2 = 1.0 - L
    ds, V3s = [], []
    for _ in range(N):
        vf = unit(rng) * L
        vss = unit(rng)
        c = np.cross(vf, vss)
        if np.linalg.norm(c) < 1e-9:
            continue
        V3 = c / np.linalg.norm(c)
        a, b = vf + V3 * z2, vf - V3 * z2
        a /= np.linalg.norm(a)
        b /= np.linalg.norm(b)
        ds.append(float(np.linalg.norm(a - b)))
        V3s.append(V3)
    ds = np.array(ds)
    V3s = np.array(V3s)

    print("=" * 76)
    print(f"(2) 固定 L = {L},  v_final 与 vss 全随机     [已实测确认]")
    print("=" * 76)
    print(f"  n = {len(ds)}")
    print(f"  间隔 min = {ds.min():.15f}")
    print(f"       max = {ds.max():.15f}")
    print(f"       std = {ds.std():.3e}")
    print(f"  理论值   = {2*z2/math.sqrt(L*L+z2*z2):.15f}")
    print(f"  V_3 覆盖范围 (证明它真的在变):")
    for k, nm in enumerate(("V3x", "V3y", "V3z")):
        print(f"      {nm} ∈ [{V3s[:,k].min():+.4f}, {V3s[:,k].max():+.4f}]")
    print(f"  相关性 (应平坦):")
    for k, nm in enumerate(("V3x", "V3y", "V3z")):
        print(f"      corr(间隔, {nm}) = {np.corrcoef(V3s[:,k], ds)[0,1]:+.4e}")
    print()
    print("  ⟹ V_3 跑遍整个单位球面, 间隔纹丝不动(1.4e-16), 相关性 ~0。")
    print("     即: 间隔只编码残差【大小】(纯 r[2] 的函数);")
    print("         朝向 V_3 才编码残差【方向】。")
    print()

    # ---------------- ③ 两条叶严格对称 ----------------
    print("=" * 76)
    print("(3) 两条叶的对称性  ⟹ 上下旋只能靠能量区分")
    print("=" * 76)
    vf = unit(rng) * L
    vss = unit(rng)
    V3 = np.cross(vf, vss)
    V3 /= np.linalg.norm(V3)
    a, b = vf + V3 * z2, vf - V3 * z2
    print(f"  |v_final + V_3·z2| = {np.linalg.norm(a):.15f}")
    print(f"  |v_final - V_3·z2| = {np.linalg.norm(b):.15f}")
    print(f"  差 = {abs(np.linalg.norm(a)-np.linalg.norm(b)):.3e}")
    print()
    print("  ⟹ 两侧模长严格相等(到机器精度)。叶面本身无法区分上下旋,")
    print("     区分只能来自两段弧上的能量/曲率(κ⁻ = κ⁺)。")
    print("     对称配置(起点在对称轴上)下两侧 κ 相同 ⟹ 简并, 这是已识别的例外。")
    print()

    # ---------------- ④ 间隔 vs L ----------------
    print("=" * 76)
    print("(4) 间隔 d(L),  L = zLength = sqrt(r[2])")
    print("=" * 76)
    print(f"    {'L':>7} {'r[2]':>10} {'d':>20}   说明")
    notes = {0.30: "轴向小, 扭曲大", 0.50: "", 0.70: "", 0.90: "",
             0.98: "", 0.999: "轴向占满, 两叶几乎重合"}
    for Lx in (0.30, 0.50, 0.70, 0.90, 0.98, 0.999):
        zx = 1.0 - Lx
        d = 2.0 * zx / math.sqrt(Lx * Lx + zx * zx)
        print(f"    {Lx:>7.3f} {Lx*Lx:>10.4f} {d:>20.12f}   {notes[Lx]}")
    print()
    print("  端点自洽: L→1 ⟹ d→0 (两叶重合); L→0 ⟹ d→2 (两叶对顶)")
    print("  且由 r[2] < 1 保证 z2 > 0, 无退化。")
    print("  这正是「单位主向量 = 轴向分量 + 扭曲分量」的两个极限。")
    print()

    print("=" * 76)
    print("结论")
    print("=" * 76)
    print("  1. V_3 ⊥ v_final, V_3 ⊥ vss          —— 叉积的直接后果")
    print("  2. 间隔 = 2z2/sqrt(L^2+z2^2), 只依赖 r[2] —— 编码残差【大小】")
    print("  3. V_3 朝向                            —— 编码残差【方向】")
    print("  4. 两条叶严格对称                      —— 上下旋只能由能量区分")
    print()
    print("  未验证: κ⁻=κ⁺ 破对称的实际表现; 两段能量 2 方程闭合的收敛性;")
    print("          ez 起步轴(MaxComponentAxis(v3))改动后的下游走向。")


if __name__ == "__main__":
    main()
