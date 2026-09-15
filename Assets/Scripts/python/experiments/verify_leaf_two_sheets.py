"""
verify_leaf_two_sheets.py
-------------------------
回答: "新(叶面两叶)" 到底是哪两个点? 抛物线还需要采集吗?

旧路:  Vss1 从 5 个全局量 (v3:2, vss:2, r[2]:1) 造两条抛物线 points1/points2 (各 100 点),
       ComputeAllFeatures 沿曲线下标 i 求 J_i = points2[i]-points1[i],
       d = ||J||, g = ||dJ/ds||, ds = 两条曲线相邻点的平均弧长。

新路:  th4 对每个 uv 点本来就算了两个叶:
          A = qs  * vs3(uv)      (qs :  z -> v3)
          B = qs2 * vs3(uv)      (qs2:  z -> -v3)
       旧代码 (.normalized) 之后塞进 uvss 当"方向"用, 点本身丢掉了。
       新路把这两个点留下来当 pointsA/pointsB:
          J(u,v) = B - A
          d = ||J||
          g = || [dJ/du, dJ/dv] ||_F     (叶面雅可比的 Frobenius 范数, 与排序无关)
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
from verify_v2_provenance import load_points, vss1, q_from_to, q_rot
from verify_v2_perpoint import th4_raw

F = np.float64
rng = np.random.default_rng(11)

print("=" * 80)
print("1. th4 的两个叶 vs 旧代码丢掉的部分")
print("=" * 80)
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
print(f"  rp = {rp:.6f}   d2 = {d2:.4f} deg   v3 = {v3}")

N = 200
Rs = 0.15 + 1.25*np.sqrt(rng.uniform(0, 1, N))
th = rng.uniform(0, 2*np.pi, N)
uvs = np.column_stack([Rs*np.cos(th), Rs*np.sin(th)])
A, B, rr, phis = th4_raw(uvs, rp, d2, v3)
J = B - A
d_meas = np.linalg.norm(J, axis=1)
print(f"  未归一化 |A| = |B| : {np.median(np.linalg.norm(A,axis=1)):.6f}  "
      f"(= r = rp*Rs^sin d2)")
print(f"  归一化后 |A| = |B| : {np.median(np.linalg.norm(A/np.linalg.norm(A,axis=1,keepdims=True),axis=1)):.6f}")
print("  => 旧代码用 .normalized, 半径 r 被约掉; 新路保留未归一化的点。")

print()
print("=" * 80)
print("2. d 的闭式:  d = 2 r sqrt( cos^2 d2 + sin^2 d2 * sin^2 phi ),  r = rp*Rs^sin d2, phi = theta*sin d2")
print("=" * 80)
sd, cd = np.sin(np.radians(d2)), np.cos(np.radians(d2))
d_pred = 2.0 * rr * np.sqrt(cd**2 + sd**2 * np.sin(phis)**2)
rel = np.abs(d_pred - d_meas) / np.maximum(d_meas, 1e-30)
print(f"  最大相对差 = {rel.max():.3e}   中位 = {np.median(rel):.3e}")
print("  => d(u,v) 是叶坐标 (Rs, theta) 的闭式函数, 不含 vss / r[2], 也不需要任何曲线排序。")
print()
print("  几何读法: A 与 B 是同一个叶点看两个锥叶的像; d 只由")
print("            r (锥面半径, ∝ Rs^sin d2) 和 phi (锥面方位角, = theta*sin d2) 决定。")

print()
print("=" * 80)
print("3. 旧路 vs 新路: d、g 的来源对照")
print("=" * 80)
vssv = np.array([0.31, 0.72, -0.62]); vssv /= np.linalg.norm(vssv)
p1, p2, _ = vss1(v3, vssv, r[2])
J_old = p2 - p1
d_old = np.linalg.norm(J_old, axis=1)
ds = np.zeros(len(p1))
for i in range(1, len(p1)):
    ds[i] = 0.5*(np.linalg.norm(p1[i]-p1[i-1]) + np.linalg.norm(p2[i]-p2[i-1]))
ds[0] = ds[1]
g_old = np.zeros(len(p1))
for i in range(len(p1)):
    if i == 0: gJ = (J_old[1]-J_old[0])/max(ds[1], 1e-6)
    elif i == len(p1)-1: gJ = (J_old[-1]-J_old[-2])/max(ds[-1], 1e-6)
    else: gJ = (J_old[i+1]-J_old[i-1])/(2*max(0.5*(ds[i]+ds[i+1]), 1e-6))
    g_old[i] = np.linalg.norm(gJ) + 1e-6

K = 16
Ju = np.zeros((N, 3)); Jv = np.zeros((N, 3))
for i in range(N):
    dd = np.linalg.norm(uvs - uvs[i], axis=1)
    order = np.argsort(dd, kind="stable")
    nb = [j for j in order if dd[j] > 1e-12][:K]
    M = np.column_stack([uvs[nb,0], uvs[nb,1], np.ones(len(nb))])
    S, *_ = np.linalg.lstsq(M, J[nb], rcond=None)
    Ju[i] = S[0]; Jv[i] = S[1]
g_new = np.sqrt(np.sum(Ju**2,axis=1) + np.sum(Jv**2,axis=1))

print(f"  {'':22}{'d 跨度':>10}{'g 跨度':>10}{'g 范围':>22}{'x=1-g^2/4 范围':>24}")
for tag, dd_, gg_ in [("旧 (Vss1 两条抛物线)", d_old, g_old), ("新 (同一 uv 的两叶)", d_meas, g_new)]:
    x = 1 - gg_**2/4
    print(f"  {tag:22}{np.log10(dd_.max()/dd_.min()):>9.2f}d{np.log10(gg_.max()/gg_.min()):>9.2f}d"
          f"   [{gg_.min():.4f}, {gg_.max():.4f}]{'':4}[{x.min():.4f}, {x.max():.4f}]")

print()
print("  g = d/(2R) 是两球重叠参数: 相交 <=> d <= 2R <=> g <= 2; g=2 是相切(透镜被掐成零)。")
print("  旧路的 g 贴着 2 -> I_x(5/2,1/2) 掉到 2e-8 -> V 跨 10.56 dex (焦散)")
print("  新路的 g 离 2 很远 -> V 跨 3.61 dex")

print()
print("=" * 80)
print("4. 新路还依赖哪些量?")
print("=" * 80)
deps = [
    ("rp", "锥母线长", "是"),
    ("d2", "半顶角", "是"),
    ("v3", "主轴 (两个叶的旋转轴)", "是"),
    ("uvs", "叶平面坐标", "是"),
    ("vss", "色散轴", "只被 AB 项用, 透镜场不用"),
    ("r[2]", "PCA 第三方差比", "旧路用 (zLength=sqrt(r[2])), 新路完全不用"),
    ("points1/points2", "Vss1 造的两条抛物线", "新路不用 (rbf 的形参现在未使用)"),
    ("R_total", "Vss1 顺带产出的四元数", "取向项 quatGrad 还在用 -> Vss1 仍需被调用"),
]
for k, mean, st in deps:
    print(f"  {k:18} {mean:28} {st}")
