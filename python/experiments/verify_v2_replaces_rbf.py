"""
verify_v2_replaces_rbf.py
-------------------------
回答: ComputeAllFeatures 塌缩成两个标量以后, 它究竟怎么替掉 RBF、又怎么进到三个势里。

A. 证明 (d, g) -> (I, V) 就是整个 ComputeAllFeatures
B. d(u,v), g(u,v) 不靠 Vss1 逐点就能拿到 (双锥两叶 + 叶面雅可比)
C. V(u,v) 是不是一个真正的场 (与叶坐标相关, 不只是 |p|^2)
D. 三个势怎么接: Vn 作系数; 概率场的平滑从 RBF 换成叶面局部二次拟合; 对比 tau 与耗时
"""
import time
import numpy as np
from scipy.special import betainc
from verify_v2_provenance import compute_all_features, vss1, load_points
from verify_v2_perpoint import th4_raw
from verify_v2_route import numerical_gradient, rbf_solve, rbf_predict, map_uv_to_s3
from verify_v2_simplified import phi_lens, fit_polynomial, tau_from_ABCD

F = np.float64
rng = np.random.default_rng(2024)


# ---------------------------------------------------- 闭式 (d,g)->(I,V)
def lens_IV(d, g):
    I = d / g
    return I, I ** 4 * phi_lens(g)


# ------------------------------------------- B. 叶面雅可比 -> d, g
def leaf_fields(uvs, rp, d2deg, v3, K=16):
    """
    双锥两叶 A(u,v), B(u,v)  ->  J = B - A
      d = ||J||
      g = ||  [dJ/du, dJ/dv] ||_F     (叶面雅可比的 Frobenius 范数, 与排序无关)
    """
    A, B, r, phis = th4_raw(uvs, rp, d2deg, v3)
    J = B - A
    d = np.linalg.norm(J, axis=1)
    N = len(uvs)
    Ju = np.zeros((N, 3)); Jv = np.zeros((N, 3))
    for i in range(N):
        dd = np.linalg.norm(uvs - uvs[i], axis=1)
        order = np.argsort(dd, kind="stable")
        nb = [j for j in order if dd[j] > 1e-12][:K]
        if len(nb) < 4:
            continue
        M = np.column_stack([uvs[nb, 0], uvs[nb, 1], np.ones(len(nb))])
        S, *_ = np.linalg.lstsq(M, J[nb], rcond=None)
        Ju[i] = S[0]; Jv[i] = S[1]
    g = np.sqrt(np.sum(Ju ** 2, axis=1) + np.sum(Jv ** 2, axis=1))
    return A, B, J, d, g, Ju, Jv


# ------------------------------------ D. 叶面局部二次拟合 (替 RBF)
def local_quad_hessian(uvs, w, K=12):
    """Savitzky-Golay 型: 每点 K 近邻拟合 w = a+bu+cv+du^2+euv+fv^2, 取 Hessian"""
    N = len(uvs)
    H = np.zeros((N, 3))
    for i in range(N):
        dd = np.linalg.norm(uvs - uvs[i], axis=1)
        order = np.argsort(dd, kind="stable")
        nb = [j for j in order if dd[j] > 1e-12][:K]
        if len(nb) < 6:
            continue
        du = uvs[nb, 0] - uvs[i, 0]
        dv = uvs[nb, 1] - uvs[i, 1]
        M = np.column_stack([np.ones(len(nb)), du, dv, du * du, du * dv, dv * dv])
        sol, *_ = np.linalg.lstsq(M, w[nb], rcond=None)
        H[i] = (2.0 * sol[3], sol[4], 2.0 * sol[5])
    return H


def stiff_from_H(uvs, H, Vn):
    u, v = uvs[:, 0], uvs[:, 1]
    Huu, Huv, Hvv = H[:, 0], H[:, 1], H[:, 2]
    s = 0.5 * (Huu * u + Huv * v) + 1j * 0.5 * (Huv * u + Hvv * v)
    return (Vn * s) if Vn is not None else s


def ab_vals(uvs, rp, d2deg, v3, vss):
    A_, B_, _, _ = th4_raw(uvs, rp, d2deg, v3)
    ua = A_ / np.linalg.norm(A_, axis=1, keepdims=True)
    ub = B_ / np.linalg.norm(B_, axis=1, keepdims=True)
    N = len(uvs)
    uvss = np.vstack([ua, ub])[:N]
    ext = uvss @ vss
    ge = numerical_gradient(uvs, ext, 12)
    return 0.5 * (ge.real - 1j * ge.imag)


if __name__ == "__main__":
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

    # ============================================================ A
    print("=" * 80)
    print("A. ComputeAllFeatures 是否只是二元映射 (d,g) -> (I,V) ?")
    print("=" * 80)
    worstI = worstV = 0.0
    for _ in range(300):
        N = 60
        # 造一对随机路径, 只要 d 和 g 对上就行
        pa = rng.normal(0, 1.0, (N, 3))
        pb = rng.normal(0, 1.0, (N, 3))
        Io, Vo = compute_all_features(pa, pb)
        J = pb - pa
        d = np.linalg.norm(J, axis=1)
        ds = np.zeros(N)
        for i in range(1, N):
            ds[i] = 0.5 * (np.linalg.norm(pa[i]-pa[i-1]) + np.linalg.norm(pb[i]-pb[i-1]))
        ds[0] = ds[1]
        eps = 1e-6; g = np.zeros(N)
        for i in range(N):
            if i == 0: gJ = (J[1]-J[0])/max(ds[1], eps)
            elif i == N-1: gJ = (J[N-1]-J[N-2])/max(ds[N-1], eps)
            else: gJ = (J[i+1]-J[i-1])/(2*max(0.5*(ds[i]+ds[i+1]), eps))
            g[i] = np.linalg.norm(gJ) + eps
        I2, V2c = lens_IV(d, g)
        worstI = max(worstI, float(np.max(np.abs(I2-Io)/np.maximum(np.abs(Io), 1e-30))))
        worstV = max(worstV, float(np.max(np.abs(V2c-Vo)/np.maximum(np.abs(Vo), 1e-30))))
    print(f"  300 对随机路径 x 60 点:  |dI/I|max = {worstI:.3e}   |dV/V|max = {worstV:.3e}")
    print("  => ComputeAllFeatures 的全部内容就是:")
    print("       d = ||J||,  g = ||dJ/ds||")
    print("       I = d/g,    V = (d/g)^4 * Phi(g)")
    print("     没有 R/R^2/x/Clamp01/不完全Beta。")

    # ============================================================ B
    print()
    print("=" * 80)
    print("B. d(u,v), g(u,v) 不用 Vss1 逐点拿到")
    print("=" * 80)
    N = 200
    Rs = 0.15 + 1.25*np.sqrt(rng.uniform(0, 1, N))
    th = rng.uniform(0, 2*np.pi, N)
    uvs = np.column_stack([Rs*np.cos(th), Rs*np.sin(th)])
    A, B, J, d, g, Ju, Jv = leaf_fields(uvs, rp, d2, v3)
    print(f"  d = |B-A|        : min {d.min():.4e}  med {np.median(d):.4e}  max {d.max():.4e}"
          f"   {np.log10(d.max()/d.min()):.2f} dex")
    print(f"  g = ||J'(u,v)||_F: min {g.min():.4e}  med {np.median(g):.4e}  max {g.max():.4e}"
          f"   {np.log10(g.max()/g.min()):.2f} dex")
    print(f"  x = 1-g^2/4      : in [{np.min(1-g*g/4):.6f}, {np.max(1-g*g/4):.6f}]"
          f"   (g>=2 的点数 {int(np.sum(g>=2))})")
    Im, Vm = lens_IV(d, g)
    print(f"  I = d/g          : min {Im.min():.4e}  med {np.median(Im):.4e}  max {Im.max():.4e}")
    print(f"  V = I^4*Phi(g)   : min {Vm.min():.4e}  med {np.median(Vm):.4e}  max {Vm.max():.4e}"
          f"   {np.log10(Vm.max()/max(Vm.min(),1e-300)):.2f} dex")
    print(f"  d 与 |p| 的 corr = {np.corrcoef(d, Rs)[0,1]:+.4f}   "
          f"g 与 |p| 的 corr = {np.corrcoef(g, Rs)[0,1]:+.4f}")
    print(f"  corr(d, theta) = {np.corrcoef(d, th)[0,1]:+.4f}   "
          f"corr(g, theta) = {np.corrcoef(g, th)[0,1]:+.4f}")
    print("  => d, g 都是从 uv 直接算出来的, 不需要 Vss1 / 两条全局抛物线 / 5 个全局量。")

    # ============================================================ C
    print()
    print("=" * 80)
    print("C. V(u,v) 是不是真正的场 ?")
    print("=" * 80)
    print(f"  corr(log V, log|p|) = {np.corrcoef(np.log(Vm), np.log(Rs))[0,1]:+.4f}")
    print(f"  corr(log V, theta)  = {np.corrcoef(np.log(Vm), th)[0,1]:+.4f}")
    print(f"  corr(log V, g)      = {np.corrcoef(np.log(Vm), g)[0,1]:+.4f}")
    # 光滑性: V 沿叶面方向的有限差分
    for name, arr in [("V", Vm), ("I", Im)]:
        gu = np.zeros(N); gv = np.zeros(N)
        for i in range(N):
            dd = np.linalg.norm(uvs - uvs[i], axis=1)
            order = np.argsort(dd, kind="stable")
            nb = [j for j in order if dd[j] > 1e-12][:12]
            M = np.column_stack([uvs[nb, 0], uvs[nb, 1], np.ones(len(nb))])
            s, *_ = np.linalg.lstsq(M, arr[nb], rcond=None)
            gu[i], gv[i] = s[0], s[1]
        grad = np.hypot(gu, gv)
        rel = grad / np.maximum(np.abs(arr), 1e-300)
        print(f"  {name}: |grad|/|value| 中位 = {np.median(rel):.4f}  "
              f"呈有限值的点 = {int(np.sum(np.isfinite(grad)))}/{N}")

    # ============================================================ D
    print()
    print("=" * 80)
    print("D. 三个势怎么接: RBF 路线 vs 叶面局部拟合 + Vn 系数")
    print("=" * 80)
    vss = np.array([0.31, 0.72, -0.62]); vss /= np.linalg.norm(vss)
    w = np.sin(2.0*uvs[:, 0])*np.cos(1.3*uvs[:, 1]) + 0.4*uvs[:, 1]

    # ---- 旧: RBF ----
    V2 = np.column_stack([Im/Im.max(), Vm/Vm.max()])
    t0 = time.perf_counter()
    X, _ = map_uv_to_s3(uvs, V2)
    ang = np.arccos(np.clip(X @ X.T, -1, 1))
    sigma = ang[np.triu_indices(N, 1)].mean()*0.8
    wc, c0, _ = rbf_solve(X, w, sigma)
    wp = rbf_predict(X, X, wc, c0, sigma)
    gw = numerical_gradient(uvs, wp, 12)
    gdu = numerical_gradient(uvs, gw.real, 12)
    gdv = numerical_gradient(uvs, gw.imag, 12)
    H_rbf = np.column_stack([gdu.real, gdu.imag, gdv.imag])
    t_rbf = time.perf_counter() - t0

    # ---- 新: 叶面局部二次拟合, 无 RBF / 无 S^3 ----
    t0 = time.perf_counter()
    H_loc = local_quad_hessian(uvs, w, 12)
    t_loc = time.perf_counter() - t0

    print(f"  Hessian 差异: |H_loc - H_rbf| 中位/|H_rbf| 中位 = "
          f"{np.median(np.abs(H_loc-H_rbf))/max(np.median(np.abs(H_rbf)),1e-30):.4f}")
    print(f"  耗时: RBF 路线 {t_rbf*1e3:.1f} ms   局部拟合 {t_loc*1e3:.1f} ms")

    lv = np.log(np.maximum(Vm, 1e-300))
    Vn = (lv - lv.min())/max(lv.max()-lv.min(), 1e-9)
    AB = ab_vals(uvs, rp, d2, v3, vss)
    fz = uvs[:, 0] + 1j*uvs[:, 1]

    print()
    print(f"  {'路线':>38} {'|stiff|':>12} {'|AB|':>12} {'Re tau':>10} {'Im tau':>10} "
          f"{'|Re|<=.5':>9} {'j':>16}")
    cases = [
        ("旧: RBF + V2 进坐标 + 0.5Hz", stiff_from_H(uvs, H_rbf, None)),
        ("新: 局部拟合 + 0.5Hz", stiff_from_H(uvs, H_loc, None)),
        ("新: 局部拟合 + Vn*0.5Hz", stiff_from_H(uvs, H_loc, Vn)),
        ("旧: RBF + Vn*0.5Hz", stiff_from_H(uvs, H_rbf, Vn)),
    ]
    for tag, stiff in cases:
        total = 6.0*(stiff + AB)
        A_, B_, C_, D_ = fit_polynomial(fz, total)
        tau, j, ratio, disc = tau_from_ABCD(A_, B_, C_, D_)
        ok = abs(tau.real) <= 0.5 + 1e-9
        print(f"  {tag:>38} {np.linalg.norm(stiff):>12.4e} {np.linalg.norm(AB):>12.4e} "
              f"{tau.real:>10.5f} {tau.imag:>10.5f} {str(ok):>9} {j.real:>16.6e}")
    print()
    print("  => stiffnessProj 的形状完全不变, 只是把'3 个势平铺相加'变成'透镜体积作系数':")
    print("       totalGrad = 6 * ( Vn ⊙ 0.5*H*z  +  Wn ⊙ AB  +  Qn ⊙ quat )")
    print("     而 H 不再需要 RBF: 概率场本身就在 uv 面上采样, 局部二次拟合直接给 Hessian。")
