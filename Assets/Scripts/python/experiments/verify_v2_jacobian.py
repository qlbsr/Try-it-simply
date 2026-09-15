"""
verify_v2_jacobian.py
---------------------
提议版 V2: 用叶映射 (uv -> 物理方向) 的雅可比/面积元, 而不是 Vss1 的透镜体积。

  th4: uv -> S^2 方向 (双锥两叶)
  在每个 uv 点用 K 近邻 lstsq 求  dX/du  (3维), dX/dv (3维)
      I_i = ||dX/du||                      空间应变 (刚度)
      V_i = ||dX/du x dX/dv||              面积元 (体积, 无 R^4 放大)

三个版本放在同一批 uv 点上比:
  (1) V-now      Vss1 全局抛物线 + ComputeAllFeatures
  (2) V-nowlog   同上, 但 V 用 log 归一化 (只改归一化)
  (3) V-jac      雅可比面积元 (逐点, 无抛物线)
"""
import numpy as np
from verify_v2_provenance import compute_all_features, vss1, load_points
from verify_v2_perpoint import th4_raw

F = np.float64


def jacobian_features(uv, X, K=12):
    """逐点 K 近邻 lstsq: X(k) = a_k*u + b_k*v + c_k"""
    N = len(uv)
    du = np.zeros((N, 3)); dv = np.zeros((N, 3))
    for i in range(N):
        d = np.linalg.norm(uv - uv[i], axis=1)
        order = np.argsort(d, kind="stable")
        nb = [j for j in order if d[j] > 1e-12][:K]
        if len(nb) < 3:
            continue
        M = np.column_stack([uv[nb, 0], uv[nb, 1], np.ones(len(nb))])
        Sol, *_ = np.linalg.lstsq(M, X[nb], rcond=None)   # (3,3)
        du[i] = Sol[0]; dv[i] = Sol[1]
    I = np.linalg.norm(du, axis=1)
    V = np.linalg.norm(np.cross(du, dv), axis=1)
    return I, V


def stats(tag, I, V, Vlog=False):
    print(f"  [{tag}]")
    if Vlog:
        lo, hi = np.log(max(V.min(), 1e-300)), np.log(V.max())
        Vn = (np.log(np.maximum(V, 1e-300)) - lo) / (hi - lo)
    else:
        Vn = V / V.max()
    In = I / I.max()
    ph = np.degrees(np.arctan2(Vn, In))
    frac_small = np.mean(ph < 1.0)
    ev2 = np.sort(np.linalg.eigvalsh(np.cov(np.column_stack([In, Vn]).T)))[::-1]
    print(f"    I 跨度 {np.log10(I.max()/max(I.min(),1e-30)):5.2f} dex   "
          f"V 跨度 {np.log10(max(V.max(),1e-300)/max(V.min(),1e-300)):5.2f} dex"
          f"   (Vlog={Vlog})")
    print(f"    V2 PCA {ev2[0]/ev2.sum():.4f}/{ev2[1]/ev2.sum():.4f}   "
          f"corr(I,V)={np.corrcoef(I,V)[0,1]:+.4f}")
    print(f"    相位范围 [{ph.min():8.3f}, {ph.max():8.3f}] deg   跨度 {ph.max()-ph.min():8.3f} deg")
    print(f"    >>> 相位 < 1 deg 的点占比 = {100*frac_small:6.2f}%   "
          f"(越高说明 V2 第二坐标被压死)")
    return In, Vn


if __name__ == "__main__":
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

    N = 200
    rng = np.random.default_rng(11)
    Rs = 0.15 + 1.25*np.sqrt(rng.uniform(0, 1, N))
    th = rng.uniform(0, 2*np.pi, N)
    uvs = np.column_stack([Rs*np.cos(th), Rs*np.sin(th)])
    print(f"points.json N={len(P)}  d2={d2:.2f}deg  rp={rp:.4f}  uv点数={N}")
    print(f"uv 近邻角中位 = {np.median(np.sort(np.linalg.norm(uvs[:,None]-uvs[None,:],axis=2),axis=1)[:,1]/np.maximum(Rs,1e-9)):.4f} rad")

    print()
    print("=" * 78)
    print("(1) V-now    Vss1 全局抛物线, V 按 max 归一化   <-- 代码现状")
    print("=" * 78)
    v0 = np.array([0.31, 0.72, -0.62]); v0 /= np.linalg.norm(v0)
    p1, p2, _ = vss1(v3, v0, r[2])
    I_now, V_now = compute_all_features(p1, p2)
    stats("V-now", I_now, V_now, Vlog=False)

    print()
    print("=" * 78)
    print("(2) V-nowlog  同一批 I/V, 只把 V 的归一化换成 log")
    print("=" * 78)
    stats("V-nowlog", I_now, V_now, Vlog=True)

    print()
    print("=" * 78)
    print("(3) V-jac    逐点雅可比面积元 (无抛物线, 无 Vss1)")
    print("=" * 78)
    A, B, rr, phis = th4_raw(uvs, rp, d2, v3)
    X = A   # 只取一个叶 (另一叶是其镜像)
    I_j, V_j = jacobian_features(uvs, X, K=12)
    stats("V-jac-max ", I_j, V_j, Vlog=False)
    print()
    stats("V-jac-log ", I_j, V_j, Vlog=True)
    print(f"    corr(I_j, Rs) = {np.corrcoef(I_j, Rs)[0,1]:+.4f}   "
          f"corr(V_j, Rs) = {np.corrcoef(V_j, Rs)[0,1]:+.4f}   "
          f"corr(V_j, I_j) = {np.corrcoef(V_j, I_j)[0,1]:+.4f}")

    print()
    print("=" * 78)
    print("汇总: 三种 V2 归一化后, 有多少逐点自由度能真正进入 ABCD")
    print("=" * 78)
    print("  V-now     相位<1deg 占比见上 => 绝大多数点相位塌缩到 0")
    print("  V-nowlog  相位铺满 45deg     => AB 项能吃到逐点 V2, 但仍是同一条曲线")
    print("  V-jac     相位铺满 45deg 且 I/V 与 uv 几何直接对应, 无 10dex 放大")
