"""
verify_parabola_vs_sheets.py
----------------------------
回答: 抛物线采集能不能被双叶替换? 两个向量携带的信息, 双叶编码进去了吗?

Vss1(v, ...) 的真实依赖:
    v_flat = v3 (字段)          <- 向量 1: 主轴
    v_2    = v  (参数 = vss)    <- 向量 2: 色散轴
    zLength = sqrt(r[2])        <- 标量

    R_ray   = FromTo(ẑ, v3)
    v_bent  = normalize(R_ray^-1 * vss)
    R_space = FromTo(v3, v_bent)
    v_final = R_space^2 * ẑ * zLength
    V_3     = normalize(v_final x vss)
    V_A = +V_3*z2,  V_D = -V_3*z2,  z2 = 1 - zLength
    v_c  = v_final - V_A - v_final_hat*z2
    v_c1 = v_final + V_A
    V_q1 = v_final + vss - V_A - v_final_hat*z2 = v_c  + vss
    V_q2 = v_final + vss + V_A                  = v_c1 + vss
"""
import sys, io
sys.path.insert(0, "experiments")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
from verify_v2_provenance import (load_points, q_from_to, q_rot,
                                  get_parabola_segment_points, compute_all_features)
from verify_v2_perpoint import th4_raw

F = np.float64


def vss1_full(v3, v, r2, num=100):
    """完整返回 Vss1 的全部中间量"""
    v_flat = np.asarray(v3, F); v_2 = np.asarray(v, F)
    R_ray = q_from_to([0, 0, 1.0], v_flat)
    R_ray_inv = np.array([R_ray[0], -R_ray[1], -R_ray[2], -R_ray[3]])
    v_bent = q_rot(R_ray_inv, v_2); v_bent = v_bent / np.linalg.norm(v_bent)
    R_space = q_from_to(v_flat, v_bent)
    R_total = np.array([0.0, 0, 0, 0])          # 不关心
    ez_bent = q_rot(R_space, [0, 0, 1.0])
    zLength = np.sqrt(r2); z2 = 1.0 - zLength
    v_final = q_rot(R_space, ez_bent) * zLength
    V_3 = np.cross(v_final, v_2); V_3 = V_3 / np.linalg.norm(V_3)
    V_A = V_3 * z2; V_D = -V_3 * z2
    vfh = v_final / np.linalg.norm(v_final)
    v_c = v_final - V_A - vfh * z2
    v_c1 = v_final + V_A
    V_q1 = v_final + v_2 - V_A - (v_final / zLength) * z2
    V_q2 = v_final + v_2 + V_A
    p1, _ = get_parabola_segment_points(v_final, V_A, V_D, num)
    p2, _ = get_parabola_segment_points(v_final, V_q1, V_q2, num)
    return dict(v_final=v_final, V_3=V_3, V_A=V_A, V_D=V_D, v_c=v_c, v_c1=v_c1,
                V_q1=V_q1, V_q2=V_q2, zLength=zLength, z2=z2,
                R_space=R_space, v_bent=v_bent, p1=p1, p2=p2)


# ============================================================== 0. 参数
P = load_points(r"C:\Users\23128\My project (2)\Assets\Resources\points.json")
mu = P.mean(0); ev, evec = np.linalg.eigh(np.cov((P - mu).T))
o = np.argsort(ev)[::-1]; ev = ev[o]; evec = evec[:, o]
r = ev / ev.sum(); rp = float(np.abs(P).mean())
pc1 = evec[:, 0].copy()
if pc1[1] < 0: pc1 = -pc1
d2 = np.degrees(np.arccos(np.clip(pc1[1], -1, 1)))
fj = np.degrees(np.arctan2(pc1[2], pc1[0]))
if fj < 0: fj += 360
v3 = np.array([np.sin(np.radians(d2))*np.cos(np.radians(fj)),
               np.cos(np.radians(d2)),
               np.sin(np.radians(d2))*np.sin(np.radians(fj))])
rng = np.random.default_rng(5)
vssA = rng.normal(size=3); vssA /= np.linalg.norm(vssA)
vssB = rng.normal(size=3); vssB /= np.linalg.norm(vssB)
print(f"v3   = {v3}")
print(f"vssA = {vssA}")
print(f"vssB = {vssB}   (与 vssA 夹角 {np.degrees(np.arccos(np.clip(vssA@vssB,-1,1))):.2f} deg)")
print(f"r[2] = {r[2]:.6f}  -> zLength = {np.sqrt(r[2]):.6f}")

# ============================================================== 1. 拼接结构
print()
print("=" * 80)
print("1. 两个向量各自扮演什么角色")
print("=" * 80)
SA = vss1_full(v3, vssA, r[2])
SB = vss1_full(v3, vssB, r[2])
print(f"  V_3 = normalize(v_final x vss):")
print(f"    A: V_3 = {SA['V_3']}   V_3·v_final = {SA['V_3']@SA['v_final']:+.3e}   V_3·vssA = {SA['V_3']@vssA:+.3e}")
print(f"    B: V_3 = {SB['V_3']}   V_3·v_final = {SB['V_3']@SB['v_final']:+.3e}   V_3·vssB = {SB['V_3']@vssB:+.3e}")
print("    => V_3 同时垂直于 v_final 与 vss; 两个向量共同定了这条轴")
print()
print(f"  第二条弦 = 第一条弦(c 到 c1)整体平移 vss ?")
print(f"    A: |V_q1 - (v_c  + vssA)| = {np.linalg.norm(SA['V_q1']-(SA['v_c'] +vssA)):.3e}")
print(f"       |V_q2 - (v_c1 + vssA)| = {np.linalg.norm(SA['V_q2']-(SA['v_c1']+vssA)):.3e}")
print(f"    B: |V_q1 - (v_c  + vssB)| = {np.linalg.norm(SB['V_q1']-(SB['v_c'] +vssB)):.3e}")
print(f"       |V_q2 - (v_c1 + vssB)| = {np.linalg.norm(SB['V_q2']-(SB['v_c1']+vssB)):.3e}")
print("    => vss 正是以【平移量】的身份进入第二条弦 —— 这就是'拼接'结构")

# ============================================================== 2. 抛物线对 vss 敏感
print()
print("=" * 80)
print("2. 抛物线对 vss 的敏感度")
print("=" * 80)
for tag, S in [("vssA", SA), ("vssB", SB)]:
    J = S['p2'] - S['p1']
    d = np.linalg.norm(J, axis=1)
    print(f"  {tag}: |弦1| = {np.linalg.norm(S['V_D']-S['V_A']):.6f}   "
          f"|弦2| = {np.linalg.norm(S['V_q2']-S['V_q1']):.6f}   "
          f"d = ||p2-p1|| med {np.median(d):.6f}  max {d.max():.6f}")
dA = np.linalg.norm(SA['p2']-SA['p1'], axis=1)
dB = np.linalg.norm(SB['p2']-SB['p1'], axis=1)
print(f"  两套 vss 下 d 的相对差 (中位) = {np.median(np.abs(dA-dB)/np.maximum(dA,1e-30)):.4f}")

# ============================================================== 3. 双叶对 vss 免疫
print()
print("=" * 80)
print("3. 双叶对 vss 免疫 —— 也就是完全没有编码第二个向量")
print("=" * 80)
N = 200
Rs = 0.15 + 1.25*np.sqrt(rng.uniform(0, 1, N)); th = rng.uniform(0, 2*np.pi, N)
uvs = np.column_stack([Rs*np.cos(th), Rs*np.sin(th)])
A_, B_, rr_, phis_ = th4_raw(uvs, rp, d2, v3)
d_sheet = np.linalg.norm(B_-A_, axis=1)
print(f"  th4 的两叶: A = qs*vs3, B = qs2*vs3;")
print(f"    qs  = FromTo(ẑ, v3)      <- 只依赖 v3")
print(f"    qs2 = FromTo(ẑ, -v3)     <- 只依赖 v3")
print(f"    vs3 = f(rp, d2, Rs, theta)  <- 不接任何向量")
print(f"  => 双叶的输入里【没有 vss】。函数签名 NappePair(uv, rp, d2, v3) 就是这个事实。")
print(f"  d_sheet = 2 r sqrt(1-(x_hat.n)^2),  n = normalize(ẑ x v3)  -> 与 vss 无关")
print(f"    实测 d_sheet: med {np.median(d_sheet):.6f}   max {d_sheet.max():.6f}")
print(f"  把 vss 换成 vssA/vssB/任意向量, d_sheet 逐位不变 (代码里根本没读它)")

# ============================================================== 4. 抛物线内部冗余
print()
print("=" * 80)
print("4. 抛物线内部是否冗余 —— 只需两端?")
print("=" * 80)
S = SA
for num in [10, 50, 100, 400]:
    p1n, _ = get_parabola_segment_points(S['v_final'], S['V_A'], S['V_D'], num)
    # 用解析式验证: 每个点都应满足 u = zLength - (zLength/z2^2) v^2
    e1 = S['v_final']/np.linalg.norm(S['v_final'])
    delta = S['V_D'] - S['V_A']
    perp = delta - (delta@e1)*e1
    e2 = perp/np.linalg.norm(perp)
    zL = np.linalg.norm(S['v_final']); z2h = np.linalg.norm(perp)*0.5
    u = p1n @ e1; v = p1n @ e2
    resid = np.abs(u - (zL - (zL/z2h**2)*v**2))
    print(f"  numPoints={num:>4}: 全部点满足 u = zLength - (zLength/z2^2) v^2 ? 最大残差 {resid.max():.3e}")
print()
print(f"  曲线的形状只由 (v_final, start, end) 决定:")
print(f"    e1 = v_final/|v_final|        <- v_final")
print(f"    e2, z2 = perp(delta)/...      <- start, end")
print(f"    v in [start·e2, end·e2]       <- start, end")
print("  => 100 个点只是这条曲线的均匀采样, 内部是【解析冗余】, 你的判断正确")
print()
print("  ComputeBendEnergy 更是只碰两端:")
print("    v_start = start·e2,  v_end = end·e2,  k = 2 zLength / z2^2")
print("    E = ∫ k^2/(1+k^2 v^2)^2.5 dv  (Simpson, v 从 v_start 到 v_end)")

# ============================================================== 5. 采样配对约定
print()
print("=" * 80)
print("5. 但 d = ||p2[i]-p1[i]|| 是按【下标】配对的, 两条曲线的 e2 不同")
print("=" * 80)
for tag, SS in [("vssA", SA), ("vssB", SB)]:
    e1 = SS['v_final']/np.linalg.norm(SS['v_final'])
    d1 = SS['V_D']-SS['V_A'];   perp1 = d1-(d1@e1)*e1; e2a = perp1/np.linalg.norm(perp1)
    d2v = SS['V_q2']-SS['V_q1']; perp2 = d2v-(d2v@e1)*e1; e2b = perp2/np.linalg.norm(perp2)
    print(f"  {tag}: 曲线1 的 e2·曲线2 的 e2 = {e2a@e2b:+.6f}   "
          f"(|差| = {np.linalg.norm(e2a-e2b):.6f})")
print("  => 两条曲线在各自平面里均匀采样, 下标 i 之间【没有几何对应关系】")
for num in [100, 200, 400]:
    p1n, _ = get_parabola_segment_points(S['v_final'], S['V_A'], S['V_D'], num)
    p2n, _ = get_parabola_segment_points(S['v_final'], S['V_q1'], S['V_q2'], num)
    dd = np.linalg.norm(p2n-p1n, axis=1)
    print(f"  numPoints={num:>4}: d 中位 {np.median(dd):.6f}   max {dd.max():.6f}")
print("  => d 的数值依赖采样数 —— 说明它混了'几何'与'采样约定'两件事")
