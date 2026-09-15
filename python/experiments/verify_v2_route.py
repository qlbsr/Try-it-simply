"""
verify_v2_route.py
------------------
追踪 V2 = (I/maxI, V/maxV) 在 legacy sjy.cs 的 rbf() 链条里到底流到了哪里。

复刻对象 (全部逐行对照 C#\legacy\sjy.cs):
  th4            line 385
  ExtractAngles  line 1472
  AnglesToS3     line 1514
  MapUVToS3      line 1529
  SphericalRBF   sjy1.cs
  NumericalGradient line 1430  (K 近邻 lstsq)
  stiffnessProj  line 1005-1016
  AB_vals        line 968-979

只做结构性判定, 不追求数值复刻整条 dn()->sx()->MapDoubleConeToLeaf() 链。
"""
import numpy as np

rng = np.random.default_rng(20260106)
F = np.float64

# ---------------------------------------------------------------- th4
def th4(V2, rp, d2deg):
    """C# th4(Vector2 uvs). V2: (N,2). 返回 (vs3_raw (N,3), vs3_unit (N,3), phis)"""
    d2 = np.deg2rad(d2deg)
    Rs = np.hypot(V2[:, 0], V2[:, 1])
    thetas = np.arctan2(V2[:, 1], V2[:, 0])
    thetas = np.where(thetas < 0, thetas + 2 * np.pi, thetas)
    r = rp * Rs ** np.sin(d2)
    phis = thetas * np.sin(d2)
    rs = r * np.sin(d2)
    zs = r * np.cos(d2)
    vs3 = np.stack([rs * np.cos(phis), rs * np.sin(phis), zs], axis=1)
    unit = vs3 / np.linalg.norm(vs3, axis=1, keepdims=True)
    return vs3, unit, phis


# ------------------------------------------------- ExtractAngles / S3
def extract_angles(uv, V2):
    """C# ExtractAngles line 1472"""
    theta1 = np.arctan2(uv[:, 1], uv[:, 0])
    theta1 = np.where(theta1 < 0, theta1 + 2 * np.pi, theta1)
    phi = np.arctan2(V2[:, 1], V2[:, 0])
    theta2 = np.arctan2(np.sin(phi).sum(), np.cos(phi).sum())
    if theta2 < 0:
        theta2 += 2 * np.pi
    meanR_V2 = np.abs(V2).sum(axis=1).mean() if False else np.linalg.norm(V2, axis=1).mean()
    r_uv = np.linalg.norm(uv, axis=1)
    ratio = np.clip(r_uv / (meanR_V2 + 1e-6), 0.1, 10.0)
    theta3 = (np.pi / 2) * (ratio - 1) / (ratio + 1)
    theta3 = np.clip(theta3, -np.pi / 2, np.pi / 2)
    return theta1, theta2, theta3, meanR_V2


def angles_to_s3(t1, t2, t3):
    """C# AnglesToS3 line 1514"""
    psi, th, ph = t3, t2, t1
    return np.stack([
        np.cos(psi) * np.cos(th) * np.cos(ph),
        np.cos(psi) * np.cos(th) * np.sin(ph),
        np.cos(psi) * np.sin(th),
        np.sin(psi),
    ], axis=-1)


def map_uv_to_s3(uv, V2):
    t1, t2, t3, mR = extract_angles(uv, V2)
    X = angles_to_s3(t1, t2, t3)
    return X, (t1, t2, t3, mR)


# ------------------------------------------------------- SphericalRBF
def rbf_kernel(X, sigma):
    G = np.clip(X @ X.T, -1, 1)
    Th = np.arccos(G)
    return np.exp(-(Th ** 2) / (2 * sigma ** 2))


def rbf_solve(X, w, sigma, lam=1e-3):
    N = len(X)
    K = rbf_kernel(X, sigma)
    A = np.zeros((N + 1, N + 1))
    A[:N, :N] = K + lam * np.eye(N)
    A[:N, N] = 1.0
    A[N, :N] = 1.0
    rhs = np.concatenate([w, [0.0]])
    sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
    return sol[:N], sol[N], K


def rbf_predict(X, Xq, wcoef, c0, sigma):
    Kq = np.exp(-(np.arccos(np.clip(Xq @ X.T, -1, 1)) ** 2) / (2 * sigma ** 2))
    return c0 + Kq @ wcoef


# ------------------------------------------- NumericalGradient (K=12)
def numerical_gradient(uv, values, K=12):
    """C# NumericalGradient line 1430: 每点取 K 近邻 (排除 dist<=1e-8), lstsq 拟合 a*u+b*v+c"""
    N = len(uv)
    g = np.zeros(N, dtype=complex)
    for i in range(N):
        d = np.linalg.norm(uv - uv[i], axis=1)
        order = np.argsort(d, kind="stable")
        nb = [j for j in order if d[j] > 1e-8][:K]
        if len(nb) < 3:
            continue
        M = np.column_stack([uv[nb, 0], uv[nb, 1], np.ones(len(nb))])
        sol, *_ = np.linalg.lstsq(M, values[nb], rcond=None)
        g[i] = complex(sol[0], sol[1])
    return g


# ------------------------------------------------------------ 自检 
def part1_normalization_kills_magnitude():
    print("=" * 74)
    print("Part 1  th4 的 .normalized 是否丢掉了 |V2| ?")
    print("=" * 74)
    N = 200
    ph = rng.uniform(0, 2 * np.pi, N)
    d2 = 35.0
    rp = 0.9647
    V2_low = np.column_stack([0.15 * np.cos(ph), 0.15 * np.sin(ph)])   # Rs = 0.15
    V2_hi = np.column_stack([1.40 * np.cos(ph), 1.40 * np.sin(ph)])    # Rs = 1.40
    raw_l, uni_l, ph_l = th4(V2_low, rp, d2)
    raw_h, uni_h, ph_h = th4(V2_hi, rp, d2)
    print(f"  |V2| = 0.15 -> raw |vs3| 中位 {np.median(np.linalg.norm(raw_l,axis=1)):.6e}"
          f"   单位化后 1.0")
    print(f"  |V2| = 1.40 -> raw |vs3| 中位 {np.median(np.linalg.norm(raw_h,axis=1)):.6e}"
          f"   单位化后 1.0")
    print(f"  raw 半径比 (1.40/0.15)^sin(d2) = {(1.40/0.15)**np.sin(np.deg2rad(d2)):.6f}")
    print(f"  >>> 单位化后两组方向最大差 = {np.abs(uni_l-uni_h).max():.3e}")
    print(f"  >>> 方位角 phis 最大差       = {np.abs(ph_l-ph_h).max():.3e}")
    print("  结论: uvss 只保留相位 thetas*sin(d2), |V2| 被完全删除。")
    return d2, rp


def part2_s3_is_two_scalars(d2, rp):
    print()
    print("=" * 74)
    print("Part 2  MapUVToS3(uv, V2) 是否只是 V2 的两个全局标量函数 ?")
    print("=" * 74)
    N = 200
    uv = rng.normal(0, 1.0, (N, 2))
    V2a = rng.uniform(0.05, 1.0, (N, 2))
    # 造一个"完全不同"的 V2, 但保持 (theta2, meanR) 一致
    mR = np.linalg.norm(V2a, axis=1).mean()
    ph_a = np.arctan2(V2a[:, 1], V2a[:, 0])
    th2_a = np.arctan2(np.sin(ph_a).sum(), np.cos(ph_a).sum())
    ph_b = rng.uniform(0, 2 * np.pi, N)          # 相位彻底打乱
    ph_b = ph_b - (np.arctan2(np.sin(ph_b).sum(), np.cos(ph_b).sum()) - th2_a)  # 但圆均值相同
    rad_b = rng.uniform(0.05, 1.0, N)
    rad_b = rad_b * (mR / rad_b.mean())          # 半径分布彻底不同, 但均值相同
    V2b = np.column_stack([rad_b * np.cos(ph_b), rad_b * np.sin(ph_b)])

    Xa, (t1a, t2a, t3a, mRa) = map_uv_to_s3(uv, V2a)
    Xb, (t1b, t2b, t3b, mRb) = map_uv_to_s3(uv, V2b)
    print(f"  V2a: theta2={t2a:.10f}  meanR={mRa:.10f}")
    print(f"  V2b: theta2={t2b:.10f}  meanR={mRb:.10f}   (逐点相位/半径都是新的)")
    print(f"  Corr(|V2a|, |V2b|) = {np.corrcoef(np.linalg.norm(V2a,axis=1), np.linalg.norm(V2b,axis=1))[0,1]:+.4f}")
    print(f"  >>> X 最大逐元素差 = {np.abs(Xa-Xb).max():.3e}")
    print("  结论: 两个数组逐点完全不同, 但 X 一模一样 => X = F(uv, theta2, meanR_V2)。")
    return uv, V2a, V2b, Xa, Xb


def part3_kernel_sensitivity(uv, V2a):
    print()
    print("=" * 74)
    print("Part 3  theta2 / meanR_V2 对球核 K 的影响有多大 ?")
    print("=" * 74)
    X0, (t1, t2, t3, mR) = map_uv_to_s3(uv, V2a)
    G = np.clip(X0 @ X0.T, -1, 1)
    ang = np.arccos(G)
    sigma = ang[np.triu_indices(len(uv), 1)].mean() * 0.8
    K0 = np.exp(-(ang ** 2) / (2 * sigma ** 2))
    print(f"  sigma = {sigma:.4f}")
    for name, fac in [("theta2 + pi/4", "t2"), ("theta2 -> pi/2", "t2max"), ("meanR x2", "mR2"), ("meanR /2", "mR05")]:
        if fac == "t2":
            Xq = angles_to_s3(t1, t2 + np.pi / 4, t3)
        elif fac == "t2max":
            Xq = angles_to_s3(t1, np.pi / 2, t3)
        else:
            scale = 2.0 if fac == "mR2" else 0.5
            ratio = np.clip(np.linalg.norm(uv, axis=1) / (mR * scale + 1e-6), 0.1, 10.0)
            t3q = np.clip((np.pi / 2) * (ratio - 1) / (ratio + 1), -np.pi / 2, np.pi / 2)
            Xq = angles_to_s3(t1, t2, t3q)
        Gq = np.clip(Xq @ Xq.T, -1, 1)
        angq = np.arccos(Gq)
        Kq = np.exp(-(angq ** 2) / (2 * sigma ** 2))
        rel = np.linalg.norm(Kq - K0) / np.linalg.norm(K0)
        print(f"  {name:16s}  ||dK||/||K|| = {rel:.6f}")
    print("  结论: 两个全局标量对核确有影响 => V2 不是完全无效, 但只有 2 个自由度。")


def part4_stiffness_is_v2_blind(uv, V2a, V2b):
    print()
    print("=" * 74)
    print("Part 4  决定性测试: V2 换成完全不同的数组, 三个势各自变化多少 ?")
    print("=" * 74)
    N = len(uv)
    w = np.sin(2.0 * uv[:, 0]) * np.cos(1.3 * uv[:, 1]) + 0.4 * uv[:, 1]   # 平滑概率场

    def chain(V2):
        X, (t1, t2, t3, mR) = map_uv_to_s3(uv, V2)
        ang = np.arccos(np.clip(X @ X.T, -1, 1))
        sigma = ang[np.triu_indices(N, 1)].mean() * 0.8
        wc, c0, K = rbf_solve(X, w, sigma)
        wp = rbf_predict(X, X, wc, c0, sigma)
        gw = numerical_gradient(uv, wp, 12)
        du = gw.real
        dv = gw.imag
        gdu = numerical_gradient(uv, du, 12)
        gdv = numerical_gradient(uv, dv, 12)
        H_uu = gdu.real
        H_uv = gdu.imag
        H_vv = gdv.imag
        stiff = 0.5 * (H_uu * uv[:, 0] + H_uv * uv[:, 1]) + \
            1j * 0.5 * (H_uv * uv[:, 0] + H_vv * uv[:, 1])
        return stiff, sigma, X

    S_a, sig_a, Xa = chain(V2a)
    S_b, sig_b, Xb = chain(V2b)
    print(f"  sigma_a={sig_a:.6f}  sigma_b={sig_b:.6f}")
    print(f"  ||X_a - X_b||_max = {np.abs(Xa-Xb).max():.3e}")
    print(f"  [stiffnessProj] ||S_a - S_b||_max / ||S_a||_med = "
          f"{np.abs(S_a-S_b).max()/np.median(np.abs(S_a)):.3e}   <-- 这就是'空间梯度'项")
    print(f"  [stiffnessProj] 完全逐位相同 ? {np.array_equal(S_a, S_b)}")

    # AB 项: 走的是 th4 -> uvss -> dot(uvss,vss)
    _, uni_a, _ = th4(V2a * np.array([1.0, 1.0]), 0.9647, 35.0)
    _, uni_b, _ = th4(V2b * np.array([1.0, 1.0]), 0.9647, 35.0)
    vss = np.array([0.31, 0.72, -0.62])
    vss = vss / np.linalg.norm(vss)
    ext_a = uni_a @ vss
    ext_b = uni_b @ vss
    ga = numerical_gradient(uv, ext_a, 12)
    gb = numerical_gradient(uv, ext_b, 12)
    ABa = 0.5 * (ga.real - 1j * ga.imag)
    ABb = 0.5 * (gb.real - 1j * gb.imag)
    print(f"  [externalPotential] ||e_a - e_b||_max = {np.abs(ext_a-ext_b).max():.6f}")
    print(f"  [AB_vals] ||AB_a - AB_b||_max / ||AB_a||_med = "
          f"{np.abs(ABa-ABb).max()/np.median(np.abs(ABa)):.3e}   <-- 这项吃到了 V2 的相位")
    print()
    print("  结论: V2 的逐点信息只进入 AB 项(且只有相位), 刚度项对 V2 逐点结构完全免疫。")


def part5_index_aliasing():
    print()
    print("=" * 74)
    print("Part 5  uvss 的下标别名: V2 的一条 lane 服务哪几个 uv 点 ?")
    print("=" * 74)
    print("  写入顺序:")
    print("    sx()      :189   uvss[0] = v3                      <-- 占掉 0 号槽")
    print("    rbf()     :927   for i in 0..L-1: th4(V2[i]) 加两条 (qs, qs2 双叶)")
    print("                     => uvss[1+2i] = lane i 叶A, uvss[2+2i] = lane i 叶B")
    print("    xl4()    :1411   uvss 已含 v3, 不追加")
    print()
    print("  读取处:")
    print("    externalPotential[i] = dot(uvss[i], vss)   i in [0, uvs.Length)")
    print("    xl1() :631  FromToRotation(points[i], uvss[i])  i in [0, points.Count)")
    print()
    L = 100
    M = 2 * L
    print(f"  若 V2.Length = L = {L}, uvs.Length = points.Count = {M}:")
    print("    uv点 i -> 正确应为 lane i//2 叶 i%2")
    lanes = [None] * M
    lanes[0] = ("v3", "-")
    k = 1
    for i in range(L):
        if k < M:
            lanes[k] = (i, "A"); k += 1
        if k < M:
            lanes[k] = (i, "B"); k += 1
    bad = [(i, lanes[i], (i // 2, i % 2)) for i in range(M)
           if lanes[i] != (i // 2, i % 2)]
    print(f"    actual:      i=0 -> {lanes[0]},  i=1 -> {lanes[1]},  i=2 -> {lanes[2]}, "
          f"i=3 -> {lanes[3]}, i=4 -> {lanes[4]}")
    print(f"    expected:    i=0 -> (0,'A'), i=1 -> (0,'B'), i=2 -> (1,'A'), i=3 -> (1,'B'), i=4 -> (2,'A')")
    print(f"    >>> 错配点数 = {len(bad)} / {M}  ({100*len(bad)/M:.1f}%)")
    print("  结论: uvss[0]=v3 让整条 lane 索引平移 1, 叶的奇偶性也被翻面。")


if __name__ == "__main__":
    d2, rp = part1_normalization_kills_magnitude()
    uv, V2a, V2b, Xa, Xb = part2_s3_is_two_scalars(d2, rp)
    part3_kernel_sensitivity(uv, V2a)
    part4_stiffness_is_v2_blind(uv, V2a, V2b)
    part5_index_aliasing()
