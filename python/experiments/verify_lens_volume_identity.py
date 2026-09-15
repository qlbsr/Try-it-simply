"""
verify_lens_volume_identity.py
------------------------------
V_arr = R^4 * I_x(5/2, 1/2),  x = 1 - d^2/(4R^2),  R = ||J|| / ||grad_v J||

要确定的三件事:
  Q1  (5/2, 1/2) 这两个参数是随意的吗 ?  ->  n 维球冠体积分数 = (1/2) I_{sin^2 a}((n+1)/2, 1/2)
  Q2  x 到底依赖什么 ? d 是否真的进去了 ?
  Q3  10 个数量级的跨度是哪一项造成的 ?
另外验证 ABCD 那边的 trace-length 关系。
"""
import numpy as np
from scipy.special import betainc
from scipy.integrate import quad
from verify_v2_provenance import vss1, load_points, compute_all_features

F = np.float64

print("=" * 78)
print("Q1  (1/2)*I_{sin^2 a}((n+1)/2, 1/2) 是不是 n 维球冠的体积分数 ?")
print("=" * 78)


def cap_frac_formula(n, alpha):
    return 0.5 * betainc((n + 1) / 2.0, 0.5, np.sin(alpha) ** 2)


def cap_frac_direct(n, alpha):
    """直接积分: Vol = S_{n-1}/(n) * int_{cos a}^{1} (1-t^2)^{(n-1)/2} dt  相对整球"""
    c = np.cos(alpha)
    f = lambda t: (1 - t * t) ** ((n - 1) / 2.0)
    num, _ = quad(f, c, 1.0, limit=200)
    den, _ = quad(f, -1.0, 1.0, limit=200)
    return num / den


print(f"{'n':>3} {'alpha':>8} {'公式 (1/2)I' :>16} {'直接积分':>16} {'差':>12}  参数 (a,b)")
for n in [2, 3, 4, 5, 6]:
    for adeg in [20, 45, 70, 110, 150]:
        a = np.radians(adeg)
        fo = cap_frac_formula(n, a)
        di = cap_frac_direct(n, a)
        print(f"{n:>3} {adeg:>8} {fo:>16.10f} {di:>16.10f} {abs(fo-di):>12.2e}"
              f"   ({n+1}/2, 1/2)")
print()
print("  => 参数 a=(n+1)/2, b=1/2 对应 n 维球冠。代码用的 (5/2, 1/2) 唯一对应 n = 4。")
print("     即 V_arr 是 4 维球(边界是 S^3)的球冠体积 —— 代码下游 MapUVToS3 正是把点送进 S^3。")

# ---------------------------------------------------------------- Q2
print()
print("=" * 78)
print("Q2  x = 1 - d^2/(4R^2) 里 d 到底进没进去 ?")
print("=" * 78)
rng = np.random.default_rng(3)
worst = 0.0
for _ in range(200000):
    d = 10 ** rng.uniform(-4, 2)
    g = 10 ** rng.uniform(-4, 2)      # g = ||grad_v J||
    R = d / g
    x_code = 1.0 - d * d / (4 * R * R)
    x_g = 1.0 - g * g / 4.0
    worst = max(worst, abs(x_code - x_g))
print(f"  随机 200000 组 (d, g):  |x_code - (1 - ||grad_v J||^2/4)| 最大 = {worst:.3e}")
print()
print("  推导: R = d/g  =>  d^2/(4R^2) = d^2 g^2 / (4 d^2) = g^2/4      d 完全约掉。")
print("  => 形状参数 x 只由 '雅可比场的导数' 决定; 分离量 d 只出现在 I_arr = d/g 里。")
print("  => V_arr = (d/g)^4 * I_{1-g^2/4}(5/2, 1/2)  =  d^4 * [ g^{-4} I_{1-g^2/4}(5/2,1/2) ]")

# ---------------------------------------------------------------- Q3
print()
print("=" * 78)
print("Q3  10 个数量级跨度来自哪里 ?")
print("=" * 78)
P = load_points(r"C:\Users\23128\My project (2)\Assets\Resources\points.json")
mu = P.mean(0); ev, evec = np.linalg.eigh(np.cov((P - mu).T))
o = np.argsort(ev)[::-1]; ev = ev[o]; evec = evec[:, o]
r = ev / ev.sum(); rp = float(np.abs(P).mean())
pc1 = evec[:, 0].copy()
if pc1[1] < 0:
    pc1 = -pc1
d2 = np.degrees(np.arccos(np.clip(pc1[1], -1, 1)))
fj = np.degrees(np.arctan2(pc1[2], pc1[0]))
if fj < 0:
    fj += 360
v3 = np.array([np.sin(np.radians(d2)) * np.cos(np.radians(fj)),
               np.cos(np.radians(d2)),
               np.sin(np.radians(d2)) * np.sin(np.radians(fj))])
v0 = np.array([0.31, 0.72, -0.62]); v0 /= np.linalg.norm(v0)
p1, p2, _ = vss1(v3, v0, r[2])

J = p2 - p1
d = np.linalg.norm(J, axis=1)
N = len(p1)
ds = np.zeros(N)
for i in range(1, N):
    ds[i] = 0.5 * (np.linalg.norm(p1[i] - p1[i - 1]) + np.linalg.norm(p2[i] - p2[i - 1]))
ds[0] = ds[1]
g = np.zeros(N)
for i in range(N):
    if i == 0:
        gJ = (J[1] - J[0]) / max(ds[1], 1e-6)
    elif i == N - 1:
        gJ = (J[N - 1] - J[N - 2]) / max(ds[N - 1], 1e-6)
    else:
        gJ = (J[i + 1] - J[i - 1]) / (2 * max(0.5 * (ds[i] + ds[i + 1]), 1e-6))
    g[i] = np.linalg.norm(gJ) + 1e-6

R = d / g
I_arr, V_arr = compute_all_features(p1, p2)
print(f"  d = ||J||       跨度 {np.log10(d.max()/d.min()):6.3f} dex   "
      f"[{d.min():.3e}, {d.max():.3e}]")
print(f"  g = ||grad_v J|| 跨度 {np.log10(g.max()/g.min()):6.3f} dex   "
      f"[{g.min():.3e}, {g.max():.3e}]")
print(f"  R = d/g         跨度 {np.log10(R.max()/R.min()):6.3f} dex   "
      f"[{R.min():.3e}, {R.max():.3e}]   4*跨度 = {4*np.log10(R.max()/R.min()):.3f} dex")
xs = np.clip(1 - g * g / 4, 0, 1)
Ix = betainc(2.5, 0.5, xs)
print(f"  x = 1-g^2/4     in [{xs.min():.4f}, {xs.max():.4f}]  "
      f"=> I_x(5/2,1/2) in [{Ix.min():.4f}, {Ix.max():.4f}]")
print(f"  半角 alpha = arccos(d/2R) = arccos(sqrt(1-x)) in "
      f"[{np.degrees(np.arccos(np.sqrt(1-xs.max()))):.2f}, "
      f"{np.degrees(np.arccos(np.sqrt(1-xs.min()))):.2f}] deg  (恒 <= 90deg)")
print(f"  V_arr           跨度 {np.log10(V_arr.max()/V_arr.min()):6.3f} dex   "
      f"[{V_arr.min():.3e}, {V_arr.max():.3e}]")
print()
print(f"  => 跨度分解: R^4 贡献 {4*np.log10(R.max()/R.min()):.2f} dex, "
      f"I_x 实际贡献 {np.log10(Ix.max()/max(Ix.min(),1e-300)):.2f} dex")
print(f"     I_x 实际范围 [{Ix.min():.3e}, {Ix.max():.3e}]   "
      f"x 最小处 = {xs.min():.4f}  <=> g = ||grad_v J|| = {g.max():.4f}")
print(f"  => V -> 0 由 x->0 触发, 即 g -> 2。而 g = d/(2R) 正是'两球重叠参数':")
print(f"     两球半径 R, 心距 d, 相交 <=> d <= 2R <=> g <= 2。g=2 是相切(焦散)。")
print(f"     实测 g in [{g.min():.4f}, {g.max():.4f}] —— 数据正贴着相切边界。")
print(f"  => 中位 V/maxV = {np.median(V_arr)/V_arr.max():.3e}   "
      f"(按 max 归一化后 70% 的点被压到相位<1deg 的原因)")

# ------------------------------------------------- ABCD trace-length
print()
print("=" * 78)
print("ABCD: 代码里的 tau 到底是什么不变量 ?")
print("=" * 78)
worst = 0.0
nz = 0
for _ in range(20000):
    A, B, C = rng.normal(size=3) + 1j * rng.normal(size=3)
    if abs(A * C) < 0.25:
        continue
    nz += 1
    disc = B * B - 4 * A * C
    ratio = (B + np.sqrt(disc)) / (B - np.sqrt(disc))
    if abs(ratio) > 1:
        ratio = 1 / ratio
    tau = np.log(ratio) / (2 * np.pi * 1j)
    lhs2 = B * B / (4 * A * C)          # 分支无关形式
    rhs2 = np.cos(np.pi * tau) ** 2
    scale = max(abs(lhs2), abs(rhs2), 1.0)
    worst = max(worst, abs(lhs2 - rhs2) / scale)
print(f"  ratio = (B+sqrt(B^2-4AC))/(B-sqrt(B^2-4AC)),  tau = log(ratio)/(2*pi*i)")
print(f"  有效样本 {nz} 组, 检验 B^2/(4AC) = cos^2(pi*tau)  最大相对偏差 = {worst:.3e}")
print()
print("  推导: (B+S)(B-S) = B^2-S^2 = 4AC  =>  ratio = (B+S)^2/(4AC),  sqrt(ratio) = e^{i*pi*tau}")
print("        cos(pi*tau) = [(B+S)/(2sqrt(AC)) + 2sqrt(AC)/(B+S)]/2 = +-B/(2sqrt(AC))")
print("  => 代码把 (B, A*C) 当成 (trace, determinant) 用, 这就是 Mobius 的 trace-multiplier 关系:")
print("        tr^2 / det = 4*cos^2(pi*tau)          (lambda = e^{2*pi*i*tau} = 乘子 ratio)")
print("     即 tau = 转移矩阵的 holonomy (平移长度 + 扭转角), 定义在 S^3 / H^3 的等距群上。")
print("     透镜体积 V2 与 tau 是同一个 Mobius 元的两个不同侧面:")
print("        tau = 复长度 (holonomy, 谱不变量)")
print("        V2  = 等距球相交的透镜体积 (几何测度不变量)")

# ---------------------------------------------- 刚度项里有没有刚度 ?
print()
print("=" * 78)
print("刚度项里到底有没有刚度 ?")
print("=" * 78)
u = np.linspace(0.2, 1.4, N)
v = np.linspace(-0.8, 0.9, N)
H_uu = rng.normal(0, 1, N)
H_uv = rng.normal(0, 1, N)
H_vv = rng.normal(0, 1, N)
stiff_raw = 0.5 * (H_uu * u + H_uv * v) + 1j * 0.5 * (H_uv * u + H_vv * v)
Vn = V_arr / V_arr.max()
stiff_w = Vn * stiff_raw
print(f"  stiffnessProj = 0.5 * H * z        (sjy.cs:1014)")
print(f"  |stiffnessProj| 中位 = {np.median(np.abs(stiff_raw)):.4e}")
print(f"  物理上刚度 = V (4维球冠体积, D ~ R^4), 故应为 V * 0.5*H*z")
print(f"  加权后 |V*0.5Hz| 中位 = {np.median(np.abs(stiff_w)):.4e}")
print(f"  两者逐点比值的跨度 = "
      f"{np.log10((Vn*np.abs(stiff_raw)).max()/max((Vn*np.abs(stiff_raw)).min(),1e-300)):.2f} dex")
print(f"  => 不乘 V, 刚度项就丢掉了最多 {np.log10(V_arr.max()/V_arr.min()):.2f} 个数量级的刚度差异。")
print("     名字叫 stiffness, 里面没有 stiffness —— 这就是它'没有意义'的直接原因。")
