"""
verify_v2_simplified.py
-----------------------
A. 闭式:  I_{1-g^2/4}(5/2, 1/2) = (12*acos(u) - 4*u*s*(5 - 2u^2)) / (6*pi),
          u = g/2, s = sqrt(1-u^2)          -- 完全初等, 不需要不完全 Beta
B. 简化后的 ComputeAllFeatures 与原版等价性
C. 把 Vn 乘到刚度项之后, ABCD -> tau -> j 怎么变
"""
import numpy as np
from scipy.special import betainc
from verify_v2_provenance import vss1, load_points, compute_all_features
from verify_v2_perpoint import th4_raw
from verify_v2_route import (map_uv_to_s3, numerical_gradient, rbf_solve,
                             rbf_predict)

F = np.float64


# ------------------------------------------------------- A. 闭式
def phi_lens(g):
    """I_{1-g^2/4}(5/2,1/2) 的初等闭式; alpha 很小时用级数避免相消"""
    g = np.asarray(g, F)
    u = np.clip(g / 2.0, 0.0, 1.0)
    alpha = np.arccos(u)
    closed = (12.0 * alpha - 8.0 * np.sin(2.0 * alpha) + np.sin(4.0 * alpha)) / (6.0 * np.pi)
    series = alpha ** 5 * (6.4 - (15360.0 / 5040.0) * alpha ** 2
                           + (258048.0 / 362880.0) * alpha ** 4) / (6.0 * np.pi)
    return np.where(alpha < 0.05, series, closed)


def phi_beta(g):
    x = np.clip(1.0 - np.asarray(g, F) ** 2 / 4.0, 0.0, 1.0)
    return betainc(2.5, 0.5, x)


# ------------------------------- B. 简化版 ComputeAllFeatures
def compute_all_features_simple(pa, pb):
    """
    与 sjy1.cs:386 等价, 但只用雅可比场 J 和它的中心差分。
      d_i = ||J_i||,  g_i = ||dJ/ds||_i
      I_i = d_i / g_i
      V_i = (d_i/g_i)^4 * Phi(g_i)
    """
    N = min(len(pa), len(pb))
    J = pb[:N] - pa[:N]
    d = np.linalg.norm(J, axis=1)
    ds = np.zeros(N)
    for i in range(1, N):
        ds[i] = 0.5 * (np.linalg.norm(pa[i]-pa[i-1]) + np.linalg.norm(pb[i]-pb[i-1]))
    if N > 1:
        ds[0] = ds[1]
    eps = 1e-6
    g = np.zeros(N)
    for i in range(N):
        if i == 0:
            step = max(ds[1], eps); gJ = (J[1]-J[0])/step
        elif i == N-1:
            step = max(ds[N-1], eps); gJ = (J[N-1]-J[N-2])/step
        else:
            step = max(0.5*(ds[i]+ds[i+1]), eps); gJ = (J[i+1]-J[i-1])/(2*step)
        g[i] = np.linalg.norm(gJ) + eps
    I = d / g
    V = I ** 4 * phi_lens(g)
    return I, V, d, g


# ------------------------------------------- C. ABCD 端到端
def fit_polynomial(f, Fv):
    """sjy.cs:1379  SVD 拟合 W = A*z + B*zbar + C*z^2 + D*zbar^2"""
    z = np.asarray(f, complex); zb = np.conj(z)
    Xm = np.column_stack([z, zb, z * z, zb * zb])
    sol, *_ = np.linalg.lstsq(Xm, Fv, rcond=None)
    return sol


def tau_from_ABCD(A, B, C, D):
    disc = B * B - 4 * A * C
    sq = np.sqrt(disc)
    ratio = (B + sq) / (B - sq)
    if abs(ratio) > 1:
        ratio = 1 / ratio
    tau = np.log(ratio) / (2 * np.pi * 1j)
    q = np.exp(2 * np.pi * 1j * tau)
    j = 1 / q + 744 + 196884 * q + 21493760 * q ** 2 + 864299970 * q ** 3
    return tau, j, ratio, disc


def full_chain(uvs, w, V2, vss, v3, rp, d2, I_arr, V_arr, mode="old", lane="pair"):
    """
    mode = 'old'  : stiffnessProj = 0.5*H*z                     (现状, sjy.cs:1014)
    mode = 'v'    : stiffnessProj = Vn * 0.5*H*z                (透镜体积作刚度)
    lane = 'pair' : uv 点 i -> lane i//2  (双锥两叶配对)
           'mod'  : uv 点 i -> lane i%L
    """
    N = len(uvs)
    L = len(V_arr)
    X, (t1, t2, t3, mR) = map_uv_to_s3(uvs, V2)
    ang = np.arccos(np.clip(X @ X.T, -1, 1))
    sigma = ang[np.triu_indices(N, 1)].mean() * 0.8
    wc, c0, K = rbf_solve(X, w, sigma)
    wp = rbf_predict(X, X, wc, c0, sigma)

    gw = numerical_gradient(uvs, wp, 12)
    gdu = numerical_gradient(uvs, gw.real, 12)
    gdv = numerical_gradient(uvs, gw.imag, 12)
    H_uu, H_uv, H_vv = gdu.real, gdu.imag, gdv.imag
    u, v = uvs[:, 0], uvs[:, 1]
    stiff = 0.5 * (H_uu*u + H_uv*v) + 1j * 0.5 * (H_uv*u + H_vv*v)

    lv = np.log(np.maximum(V_arr, 1e-300))
    VnL = (lv - lv.min()) / max(lv.max() - lv.min(), 1e-9)
    if lane == "pair":
        Vn = VnL[np.minimum(np.arange(N) // 2, L - 1)]
    else:
        Vn = VnL[np.arange(N) % L]

    if mode == "v":
        stiff = Vn * stiff
    else:
        Vn = np.ones(N)

    # AB_vals: th4 -> uvss -> dot(uvss, vss)
    A_, B_, rr, phis = th4_raw(uvs, rp, d2, v3)
    uvss = np.vstack([A_ / np.linalg.norm(A_, axis=1, keepdims=True),
                      B_ / np.linalg.norm(B_, axis=1, keepdims=True)])
    uvss = uvss[:N]                     # 与原代码 externalPotential[i]=dot(uvss[i],vss) 对齐
    ext = uvss @ vss
    ge = numerical_gradient(uvs, ext, 12)
    AB = 0.5 * (ge.real - 1j * ge.imag)

    quat = np.zeros(N, dtype=complex)   # 单独测刚度项, 取向项置零
    total = 6.0 * (stiff + AB + quat)
    fz = uvs[:, 0] + 1j * uvs[:, 1]
    A, B, C, D = fit_polynomial(fz, total)
    tau, j, ratio, disc = tau_from_ABCD(A, B, C, D)
    return dict(A=A, B=B, C=C, D=D, tau=tau, j=j, ratio=ratio, disc=disc,
                total=total, stiff=stiff, AB=AB, Vn=Vn,
                ns=float(np.linalg.norm(stiff)), na=float(np.linalg.norm(AB)))


if __name__ == "__main__":
    # ---------------- A ----------------
    print("=" * 78)
    print("A. 闭式 Phi(g) 与 不完全Beta 对照")
    print("=" * 78)
    gs = np.concatenate([np.linspace(1e-6, 0.1, 5), np.linspace(0.2, 1.9, 10),
                         np.linspace(1.95, 1.9999, 4)])
    print(f"{'g':>10} {'闭式':>18} {'BetaIncomplete':>18} {'相对差':>12}")
    worst = 0.0
    for g in gs:
        a = phi_lens(g); b = phi_beta(g)
        rel = abs(a-b)/max(b, 1e-300)
        worst = max(worst, rel)
        print(f"{g:>10.5f} {a:>18.12e} {b:>18.12e} {rel:>12.2e}")
    print(f"  => 最大相对差 = {worst:.3e}   (纯浮点)")
    print("  => 完全不依赖不完全 Beta / MathNet / 连分数 / Lanczos Gamma。")

    # ---------------- B ----------------
    print()
    print("=" * 78)
    print("B. 简化版与原版 ComputeAllFeatures 等价性")
    print("=" * 78)
    P = load_points(r"C:\Users\23128\My project (2)\Assets\Resources\points.json")
    mu = P.mean(0); ev, evec = np.linalg.eigh(np.cov((P-mu).T))
    o = np.argsort(ev)[::-1]; ev = ev[o]; evec = evec[:, o]
    r = ev/ev.sum(); rp = float(np.abs(P).mean())
    pc1 = evec[:, 0].copy()
    if pc1[1] < 0: pc1 = -pc1
    d2 = np.degrees(np.arccos(np.clip(pc1[1], -1, 1)))
    fj = np.degrees(np.arctan2(pc1[2], pc1[0]))
    if fj < 0: fj += 360
    v3 = np.array([np.sin(np.radians(d2))*np.cos(np.radians(fj)),
                   np.cos(np.radians(d2)),
                   np.sin(np.radians(d2))*np.sin(np.radians(fj))])
    v0 = np.array([0.31, 0.72, -0.62]); v0 /= np.linalg.norm(v0)
    p1, p2, _ = vss1(v3, v0, r[2])
    I_o, V_o = compute_all_features(p1, p2)
    I_s, V_s, dd, gg = compute_all_features_simple(p1, p2)
    print(f"  I 最大相对差 = {np.abs(I_s-I_o).max()/np.abs(I_o).max():.3e}")
    print(f"  V 最大相对差 = {np.abs(V_s-V_o).max()/np.abs(V_o).max():.3e}")
    print(f"  (原版 V 里有 2.1e-11 的极小值, 相对差在那里会被放大; 比绝对量)")
    print(f"  |V_s - V_o| 最大 = {np.abs(V_s-V_o).max():.3e}   "
          f"V_o 最大 = {V_o.max():.3e}")
    print(f"  逐点 I 相同? {np.allclose(I_s, I_o)}   逐点 V 相同? {np.allclose(V_s, V_o)}")

    # ---------------- C ----------------
    print()
    print("=" * 78)
    print("C. Vn 乘到刚度项后 ABCD -> tau -> j 的变化")
    print("=" * 78)
    rng = np.random.default_rng(11)
    N = 200
    Rs = 0.15 + 1.25*np.sqrt(rng.uniform(0, 1, N))
    th = rng.uniform(0, 2*np.pi, N)
    uvs = np.column_stack([Rs*np.cos(th), Rs*np.sin(th)])
    w = np.sin(2.0*uvs[:, 0])*np.cos(1.3*uvs[:, 1]) + 0.4*uvs[:, 1]
    V2 = np.column_stack([I_s/I_s.max(), V_s/V_s.max()])
    vss = v0

    print(f"  {'模式':>26} {'|stiff|':>12} {'|AB|':>12} {'Re tau':>10} {'Im tau':>10} "
          f"{'|Re|<=.5':>9} {'j':>16}")
    res = {}
    combos = [("old", "pair"), ("v   (Vn*0.5Hz)", "pair"), ("v   (Vn*0.5Hz, i%L)", "mod")]
    for tag, mode, lane in [(c[0], "old" if c[0] == "old" else "v", c[1]) for c in combos]:
        R_ = full_chain(uvs, w, V2, vss, v3, rp, d2, I_s, V_s, mode=mode, lane=lane)
        res[tag] = R_
        ok = abs(R_['tau'].real) <= 0.5 + 1e-9
        print(f"  {tag:>26} {R_['ns']:>12.4e} {R_['na']:>12.4e} "
              f"{R_['tau'].real:>10.5f} {R_['tau'].imag:>10.5f} {str(ok):>9} "
              f"{R_['j'].real:>16.6e}")
    print()
    for tag in list(res)[1:]:
        dt = abs(res[tag]['tau'] - res['old']['tau'])
        print(f"  |tau({tag}) - tau(old)| = {dt:.6f}   "
              f"|j ratio| = {abs(res[tag]['j']/res['old']['j']):.6f}   "
              f"|stiff|/|AB| 由 {res['old']['ns']/max(res['old']['na'],1e-300):.4f} "
              f"-> {res[tag]['ns']/max(res[tag]['na'],1e-300):.4f}")
    print()
    print(f"  Vn 的范围 [{res['v   (Vn*0.5Hz)']['Vn'].min():.4f}, "
          f"{res['v   (Vn*0.5Hz)']['Vn'].max():.4f}], "
          f"均值 {res['v   (Vn*0.5Hz)']['Vn'].mean():.4f}")
    print()
    print("  注: w 用合成平滑场 (无真实 probs), 这里测的是灵敏度与量级, 不是真实 tau。")
