"""
verify_constant_part.py
-----------------------
把 ∂U/∂z = A z + B z̄ + C z² + D z̄² + 高阶 里的"常数部分"单独拎出来:

  代码里的刚度投影 (sjy.cs:1009 原式 / v2sjy.cs StiffnessProjectionFromH):
      stiffProj(z) = 0.5*[ (H_uu*u + H_uv*v) + i*(H_uv*u + H_vv*v) ]

  要证明: 它恒等于 A z + B z̄ 的形式, 即
      stiffProj(z) = alpha*z + beta*z̄
      alpha = (H_uu + H_vv)/4
      beta  = (H_uu - H_vv + 2i*H_uv)/4
  并且 |alpha|^2 - |beta|^2 = det(H)/4

  推论: 若 H 与点无关 (常数 Hessian), 则 C = D = 0 (逐位). 于是
        C, D 两个通道只能由 H 的"空间变化"以及 AB/quat 两个势产生。
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
from verify_v2_simplified import phi_lens, fit_polynomial, tau_from_ABCD
from verify_v2_provenance import load_points, vss1
from verify_v2_perpoint import th4_raw
from verify_v2_replaces_rbf import leaf_fields, local_quad_hessian, lens_IV
from verify_v2_route import numerical_gradient

rng = np.random.default_rng(11)
F = np.float64


def stiff_proj(uvs, Huu, Huv, Hvv, coef=None):
    """与 C# StiffnessProjectionFromH 逐字对应"""
    u, v = uvs[:, 0], uvs[:, 1]
    re = 0.5 * (Huu * u + Huv * v)
    im = 0.5 * (Huv * u + Hvv * v)
    s = re + 1j * im
    return s if coef is None else coef * s


print("=" * 82)
print("1. 恒等式  stiffProj(z) = alpha*z + beta*zbar")
print("=" * 82)
worst_id, worst_det = 0.0, 0.0
for _ in range(200000):
    Huu, Huv, Hvv = rng.normal(size=3) * 3.0
    z = rng.normal() + 1j * rng.normal()
    u, v = z.real, z.imag
    S = 0.5 * ((Huu * u + Huv * v) + 1j * (Huv * u + Hvv * v))
    alpha = (Huu + Hvv) / 4.0
    beta = (Huu - Hvv + 2j * Huv) / 4.0
    worst_id = max(worst_id, abs(S - (alpha * z + beta * z.conjugate())))
    det = Huu * Hvv - Huv * Huv
    worst_det = max(worst_det, abs((abs(alpha) ** 2 - abs(beta) ** 2) - det / 4.0))
print(f"   200000 组随机 (H_uu,H_uv,H_vv,z):")
print(f"      |S - (alpha*z + beta*zbar)| 最大 = {worst_id:.3e}")
print(f"      ||alpha|^2 - |beta|^2 - det(H)/4| 最大 = {worst_det:.3e}")
print()
print("   推导: u=(z+z̄)/2, v=(z-z̄)/(2i), 代入 0.5[(H_uu+iH_uv)u + (H_uv+iH_vv)v]")
print("         => (1/4)[ (H_uu+H_vv) z + (H_uu-H_vv+2iH_uv) z̄ ]")
print("   即  alpha = tr(H)/4  (= Laplacian/4),  beta = (H_uu-H_vv+2iH_uv)/4")
print("   而 |alpha|^2-|beta|^2 = det(H)/4  ——  共形/反共形分解的 Jacobian 行列式")

print()
print("=" * 82)
print("2. 常数 Hessian 时 C = D = 0 ?")
print("=" * 82)
N = 200
Rs = 0.15 + 1.25 * np.sqrt(rng.uniform(0, 1, N))
th = rng.uniform(0, 2 * np.pi, N)
uvs = np.column_stack([Rs * np.cos(th), Rs * np.sin(th)])
fz = uvs[:, 0] + 1j * uvs[:, 1]

for tag, cm in [("小", 0.3), ("中", 3.0), ("大", 50.0)]:
    Huu, Huv, Hvv = cm * 1.0, cm * 0.4, -cm * 0.7        # 常数 H
    S = stiff_proj(uvs, np.full(N, Huu), np.full(N, Huv), np.full(N, Hvv))
    A, B, C, D = fit_polynomial(fz, S)
    alpha = (Huu + Hvv) / 4.0
    beta = (Huu - Hvv + 2j * Huv) / 4.0
    print(f"   H 量级 {tag:>2} (Huu={Huu:6.2f}, Huv={Huv:5.2f}, Hvv={Hvv:6.2f}):")
    print(f"     拟合 A = {A:.6e}   (解析 alpha = {alpha:.6e})   差 {abs(A-alpha):.2e}")
    print(f"     拟合 B = {B:.6e}   (解析 beta  = {beta:.6e})   差 {abs(B-beta):.2e}")
    print(f"     拟合 C = {C:.3e}   拟合 D = {D:.3e}   <- 应为 0")
    print(f"     |C|/max(|A|,|B|) = {abs(C)/max(abs(A),abs(B)):.3e}   "
          f"|D|/max(|A|,|B|) = {abs(D)/max(abs(A),abs(B)):.3e}")

print()
print("=" * 82)
print("3. 真实数据: 常数部分 vs 变化部分 各自进了哪些通道")
print("=" * 82)
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
vssv = np.array([0.31, 0.72, -0.62]); vssv /= np.linalg.norm(vssv)

A_, B_, J, d, g, Ju, Jv = leaf_fields(uvs, rp, d2, v3)
I_, V_ = lens_IV(d, g)
lv = np.log(np.maximum(V_, 1e-300))
Vn = (lv - lv.min()) / max(lv.max() - lv.min(), 1e-9)

w = np.sin(2.0*uvs[:, 0])*np.cos(1.3*uvs[:, 1]) + 0.4*uvs[:, 1]
H = local_quad_hessian(uvs, w, 12)
Huu, Huv, Hvv = H[:, 0], H[:, 1], H[:, 2]
print(f"   H_uu in [{Huu.min():.4f}, {Huu.max():.4f}]  std={Huu.std():.4f}   "
      f"均值 {Huu.mean():.4f}  -> std/|mean| = {Huu.std()/max(abs(Huu.mean()),1e-30):.4f}")

def fitit(Fv, tag):
    A, B, C, D = fit_polynomial(fz, Fv)
    nrm = np.linalg.norm(np.column_stack([A, B, C, D]))
    print(f"   {tag:38} A={A:9.4f}  B={B:9.4f}  C={C:9.4f}  D={D:9.4f}")
    print(f"   {'':38} |C|/总={abs(C)/nrm:.4f}  |D|/总={abs(D)/nrm:.4f}")
    return A, B, C, D

print()
print("   三个输入分别是:")
S_var = stiff_proj(uvs, Huu, Huv, Hvv, Vn)                 # 逐点 H 乘逐点 Vn
S_const = stiff_proj(uvs, np.full(N, Huu.mean()), np.full(N, Huv.mean()),
                     np.full(N, Hvv.mean()), Vn)           # H 取均值, Vn 仍逐点
S_cVc = stiff_proj(uvs, np.full(N, Huu.mean()), np.full(N, Huv.mean()),
                   np.full(N, Hvv.mean()), np.full(N, Vn.mean()))  # H, Vn 都取常数

fitit(S_var,   "逐点 H * 逐点 Vn       (真实刚度项)")
fitit(S_const, "H 取均值 * 逐点 Vn")
fitit(S_cVc,   "H 取均值 * Vn 取均值    (全常数)")
print()
print("   => 前两项的 C,D 都由 H 的空间变化 + Vn 的空间变化产生;")
print("      第三项完全落进 A,B, C=D=0。")

print()
print("=" * 82)
print("4. 判别式的三个候选 (A,B,C,D 是四个通道时)")
print("=" * 82)
ABv = None
A_, B_, rr, _ = th4_raw(uvs, rp, d2, v3)
ua = A_ / np.linalg.norm(A_, axis=1, keepdims=True)
ub = B_ / np.linalg.norm(B_, axis=1, keepdims=True)
ext = np.vstack([ua, ub])[:N] @ vssv
ge = numerical_gradient(uvs, ext, 12)
AB = 0.5 * (ge.real - 1j * ge.imag)
tot = 6.0 * (S_var + AB)
A, B, C, D = fit_polynomial(fz, tot)
print(f"   实测 A={A:.6f}  B={B:.6f}  C={C:.6f}  D={D:.6f}")
print()
cands = [
    ("代码现状   disc = B^2 - 4AC", B*B - 4*A*C),
    ("特征多项式 disc = (A-D)^2 + 4BC", (A-D)**2 + 4*B*C),
    ("同价形式   disc = (A+D)^2 - 4(AD-BC)", (A+D)**2 - 4*(A*D - B*C)),
]
for tag, disc in cands:
    sq = np.sqrt(disc) if abs(disc) > 0 else 0j
    if abs(B + sq) > 1e-300 and abs(B - sq) > 1e-300:
        ratio = (B + sq) / (B - sq)
        if abs(ratio) > 1: ratio = 1/ratio
        tau = np.log(ratio) / (2*np.pi*1j)
        q = np.exp(2*np.pi*1j*tau)
        j = 1/q + 744 + 196884*q + 21493760*q**2 + 864299970*q**3
        print(f"   {tag:36} disc={disc:12.6f}  |ratio|={abs(ratio):.6f}  "
              f"tau={tau.real:+.5f}{tau.imag:+.5f}i  |q|={abs(q):.5f}")
    else:
        print(f"   {tag:36} disc={disc:12.6f}   (分母为零, 无法取 ratio)")
print()
print("   注: 只有 '特征多项式' 那一条是 [[A,B],[C,D]] 这个矩阵的天然不变量,")
print("       也就是 Möbius 元的 tr^2-4det = (A-D)^2+4BC。")
print("       代码用的 B^2-4AC 只碰了 A,B,C 三个通道, D 完全没进入。")
