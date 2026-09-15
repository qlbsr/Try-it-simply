"""
verify_v2_perpoint.py
---------------------
对照组: 把 ComputeAllFeatures 的输入从"一条全局抛物线 + 数组下标"换成
"同一个 uv 点在双锥两个叶上的像"之后, V2 是否变成真正的逐点空间量。

对比两个版本:
  [V-now]  points1/points2 = Vss1 生成的两条抛物线 (由 v3, vss, r[2] 决定)
  [V-new]  pointsA_i/pointsB_i = th4(uv_i) 的两个叶方向 (qs, qs2), 不归一化

两者都用 sjy1.cs 的 LocalLensVolumeExtractor.ComputeAllFeatures 原样计算。
"""
import numpy as np
from scipy.special import betainc
from verify_v2_provenance import (compute_all_features, q_from_to, q_rot,
                                  vss1, load_points, q_axis_angle)

F = np.float64


def th4_raw(uv_pts, rp, d2deg, v3):
    """C# th4 (sjy.cs:385) 但保留半径, 返回两个叶上的点 (A=qs叶, B=qs2叶)"""
    d2 = np.radians(d2deg)
    Rs = np.linalg.norm(uv_pts, axis=1)
    thetas = np.arctan2(uv_pts[:, 1], uv_pts[:, 0])
    thetas = np.where(thetas < 0, thetas + 2*np.pi, thetas)
    r = rp * Rs ** np.sin(d2)
    phis = thetas * np.sin(d2)
    rs = r * np.sin(d2)
    zs = r * np.cos(d2)
    vs3 = np.stack([rs*np.cos(phis), rs*np.sin(phis), zs], axis=1)
    qs = q_from_to([0, 0, 1.0], v3/np.linalg.norm(v3))
    qs2 = q_from_to([0, 0, 1.0], -v3/np.linalg.norm(v3))
    A = np.array([q_rot(qs, p) for p in vs3])
    B = np.array([q_rot(qs2, p) for p in vs3])
    return A, B, r, phis


def is_planar(P):
    return np.linalg.svd(P - P.mean(0), compute_uv=False)


def report(tag, I, V, extra=""):
    print(f"  [{tag}]")
    print(f"    I : min {I.min():.4e}  med {np.median(I):.4e}  max {I.max():.4e}"
          f"   量级跨度 {np.log10(I.max()/max(I.min(),1e-30)):.2f} dex")
    print(f"    V : min {V.min():.4e}  med {np.median(V):.4e}  max {V.max():.4e}"
          f"   量级跨度 {np.log10(V.max()/max(V.min(),1e-30)):.2f} dex")
    In, Vn = I/I.max(), V/V.max()
    V2 = np.column_stack([In, Vn])
    ev2 = np.sort(np.linalg.eigvalsh(np.cov(V2.T)))[::-1]
    ph = np.degrees(np.arctan2(Vn, In))
    print(f"    V2 点云 PCA 方差比 {ev2[0]/ev2.sum():.6f} / {ev2[1]/ev2.sum():.6f}"
          f"   corr(I,V) = {np.corrcoef(I,V)[0,1]:+.4f}")
    print(f"    相位 atan2(V/maxV, I/maxI) 范围 = [{ph.min():.2f}, {ph.max():.2f}] deg"
          f"   跨度 {ph.max()-ph.min():.2f} deg")
    if extra:
        print(f"    {extra}")


if __name__ == "__main__":
    P = load_points(r"C:\Users\23128\My project (2)\Assets\Resources\points.json")
    mu = P.mean(0); ev, evec = np.linalg.eigh(np.cov((P-mu).T))
    o = np.argsort(ev)[::-1]; ev = ev[o]; evec = evec[:, o]
    r = ev/ev.sum()
    rp = float(np.abs(P).mean())
    pc1 = evec[:, 0].copy()
    if pc1[1] < 0: pc1 = -pc1
    d2 = np.degrees(np.arccos(np.clip(pc1[1], -1, 1)))
    fj = np.degrees(np.arctan2(pc1[2], pc1[0]))
    if fj < 0: fj += 360
    v3 = np.array([np.sin(np.radians(d2))*np.cos(np.radians(fj)),
                   np.cos(np.radians(d2)),
                   np.sin(np.radians(d2))*np.sin(np.radians(fj))])
    print(f"points.json: N={len(P)}  r={r}  rp={rp:.4f}  d2={d2:.2f}deg")

    # uvs: 叶平面上的一团点 (半径/角度分布仿照真实 uvs)
    N = 200
    rng = np.random.default_rng(11)
    Rs = 0.15 + 1.25 * np.sqrt(rng.uniform(0, 1, N))
    th = rng.uniform(0, 2*np.pi, N)
    uvs = np.column_stack([Rs*np.cos(th), Rs*np.sin(th)])

    # ---------- V-now: 全局抛物线版 ----------
    print()
    print("=" * 78)
    print("V-now   两条全局抛物线 (Vss1), 100 点/条")
    print("=" * 78)
    v0 = np.array([0.31, 0.72, -0.62]); v0 /= np.linalg.norm(v0)
    p1, p2, dg = vss1(v3, v0, r[2])
    I_now, V_now = compute_all_features(p1, p2)
    report("V-now", I_now, V_now,
           f"曲线平面性(第3奇异值) p1={is_planar(p1)[2]:.2e} p2={is_planar(p2)[2]:.2e}")
    print(f"    points1/points2 由 5 个全局数决定, 与 {N} 个 uv 点无几何对应关系")

    # ---------- V-new: 双锥两叶逐点版 ----------
    print()
    print("=" * 78)
    print("V-new   同一 uv 点在双锥两叶上的像 (逐点), 不归一化")
    print("=" * 78)
    A, B, rr, phis = th4_raw(uvs, rp, d2, v3)
    print(f"    两叶间距 |B-A| : min {np.linalg.norm(B-A,axis=1).min():.4e}  "
          f"med {np.median(np.linalg.norm(B-A,axis=1)):.4e}  "
          f"max {np.linalg.norm(B-A,axis=1).max():.4e}")
    print(f"    理论 |B-A| = 2 r sin(d2), r = rp*Rs^sin(d2): "
          f"med {np.median(2*rr*np.sin(np.radians(d2))):.4e}")
    I_new, V_new = compute_all_features(A, B)
    report("V-new", I_new, V_new,
           f"V 与 Rs 的 corr = {np.corrcoef(V_new, Rs)[0,1]:+.4f}, "
           f"与 |B-A| 的 corr = {np.corrcoef(V_new, np.linalg.norm(B-A,axis=1))[0,1]:+.4f}")
    print(f"    相位跨度 {np.degrees(np.arctan2(V_new/V_new.max(), I_new/I_new.max()).max()-np.arctan2(V_new/V_new.max(), I_new/I_new.max()).min()):.2f} deg"
          f"  (对比 V-now 的 45 deg 上限)")

    # ---------- 稳定性: 扰动 v3 / r[2] ----------
    print()
    print("=" * 78)
    print("稳定性: 同样的扰动下, 两个版本的 V2 变化多少")
    print("=" * 78)
    base_now = np.column_stack([I_now/I_now.max(), V_now/V_now.max()])
    base_new = np.column_stack([I_new/I_new.max(), V_new/V_new.max()])
    ax = np.cross(v3, [0, 0, 1.0]); ax /= np.linalg.norm(ax)
    for name, dvs, dr2 in [("v3 转 0.5deg", 0.5, 1.0), ("r[2] 扰动 1%", 0.0, 1.01)]:
        v3p = q_rot(q_axis_angle(np.radians(dvs), ax), v3) if dvs else v3
        r2p = r[2]*dr2
        q1, q2, _ = vss1(v3p, v0, r2p)
        I1, V1 = compute_all_features(q1, q2)
        c1 = np.column_stack([I1/I1.max(), V1/V1.max()])
        rel_now = np.linalg.norm(c1-base_now)/np.linalg.norm(base_now)
        d2p = np.degrees(np.arccos(np.clip(v3p[1], -1, 1)))
        fjp = np.degrees(np.arctan2(v3p[2], v3p[0]))
        if fjp < 0: fjp += 360
        v3pp = np.array([np.sin(np.radians(d2p))*np.cos(np.radians(fjp)),
                         np.cos(np.radians(d2p)),
                         np.sin(np.radians(d2p))*np.sin(np.radians(fjp))])
        A1, B1, _, _ = th4_raw(uvs, rp, d2p, v3pp)
        I2, V2b = compute_all_features(A1, B1)
        c2 = np.column_stack([I2/I2.max(), V2b/V2b.max()])
        rel_new = np.linalg.norm(c2-base_new)/np.linalg.norm(base_new)
        print(f"  {name:16s}  V-now ||dV2||/||V2|| = {rel_now:12.6f}    "
              f"V-new = {rel_new:12.6f}")
    print()
    print("  V-now 对 r[2] 的依赖是灾难性的(见上一脚本 11.8), V-new 只经由 d2=acos(v3.y)。")
