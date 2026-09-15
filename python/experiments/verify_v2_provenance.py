"""
verify_v2_provenance.py
-----------------------
V2 = (I/maxI, V/maxV) 到底"由什么决定"、有几个自由度。

逐行复刻 C#\legacy\sjy.cs:
  Vss1                       line 659-714
  FindIntersections          line 777
  GetVertexFromFourPoints    line 716
  GetParabolaSegmentPoints   (line ~840)
sjy1.cs:
  LocalLensVolumeExtractor.ComputeAllFeatures  line 386

结论要检验三件事:
  Q1  points1/points2 是不是只由 (v3, v, r[2]) 这几个全局量决定 ?
  Q2  V2 这个 (N,2) 数组在它自己的 2 维空间里是不是退化成 1 维 ?
  Q3  V2 的逐点变化是不是只是"曲线参数 i/N", 与 uv 点毫无对应 ?
"""
import json
import numpy as np
from scipy.special import betainc

rng = np.random.default_rng(7)
F = np.float64

# ----------------------------------------------------------- quaternion
def q_from_to(a, b):
    """Unity Quaternion.FromToRotation (shortest arc)"""
    a = np.asarray(a, F); b = np.asarray(b, F)
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    c = np.cross(a, b); d = float(np.dot(a, b))
    if np.linalg.norm(c) < 1e-12:
        if d > 0:
            return np.array([1.0, 0, 0, 0])
        ax = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
        ax = np.cross(a, ax); ax = ax / np.linalg.norm(ax)
        return np.array([0.0, *ax])
    q = np.array([1.0 + d, c[0], c[1], c[2]])
    return q / np.linalg.norm(q)


def q_mul(p, q):
    w1, x1, y1, z1 = p; w2, x2, y2, z2 = q
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2])


def q_rot(q, v):
    w, x, y, z = q
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]])
    return R @ np.asarray(v, F)


def q_axis_angle(angle, axis):
    axis = np.asarray(axis, F); n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.array([1.0, 0, 0, 0])
    axis = axis / n
    h = angle * 0.5
    return np.array([np.cos(h), *(axis * np.sin(h))])


def q_angle(a, b):
    d = abs(float(np.clip(np.dot(a, b), -1, 1)))
    return 2.0 * np.arccos(d)


# --------------------------------------------------- 几何 (Vss1 及其子程序)
def get_parabola_segment_points(v_final, start, end, num=100):
    zLength = np.linalg.norm(v_final)
    e1 = v_final / zLength
    delta = np.asarray(end, F) - np.asarray(start, F)
    perp = delta - np.dot(delta, e1) * e1
    if np.dot(perp, perp) < 1e-9:
        t = np.linspace(0, 1, num)[:, None]
        return start + t * (end - start), 0.0
    e2 = perp / np.linalg.norm(perp)
    z2 = np.linalg.norm(perp) * 0.5
    v_start = float(np.dot(start, e2)); v_end = float(np.dot(end, e2))
    minV, maxV = min(v_start, v_end), max(v_start, v_end)
    t = np.linspace(0, 1, num)
    vc = minV + (maxV - minV) * t
    uc = zLength - (zLength / (z2 * z2)) * vc * vc
    pts = uc[:, None] * e1[None, :] + vc[:, None] * e2[None, :]
    return pts, 0.0


def find_intersections(v_final, V_A, V_3, v_c, v_c1):
    zLength = np.linalg.norm(v_final)
    z2 = np.linalg.norm(V_A)
    e1 = v_final / zLength
    e2 = V_3 / np.linalg.norm(V_3)
    u0, v0 = float(np.dot(v_c, e1)), float(np.dot(v_c, e2))
    u1, v1 = float(np.dot(v_c1, e1)), float(np.dot(v_c1, e2))
    du, dv = u1 - u0, v1 - v0
    k = zLength / (z2 * z2)
    A = k * dv * dv
    B = k * 2 * v0 * dv + du
    C = k * v0 * v0 + u0 - zLength
    disc = B * B - 4 * A * C
    svals = []
    if disc >= 0 and abs(A) > 1e-30:
        sq = np.sqrt(disc)
        for s in ((-B - sq) / (2 * A), (-B + sq) / (2 * A)):
            if 0 <= s <= 1:
                svals.append(s)
        svals.sort()
    if len(svals) >= 2:
        return (v_c + svals[0] * (v_c1 - v_c), v_c + svals[1] * (v_c1 - v_c))
    return (np.asarray(v_c, F), np.asarray(v_c1, F))


def get_vertex_from_four_points(p1, p2, p3, p4):
    p1, p2, p3, p4 = map(lambda z: np.asarray(z, F), (p1, p2, p3, p4))
    origin = p1
    xa = p2 - p1
    if np.dot(xa, xa) < 1e-8:
        return np.zeros(3)
    xa = xa / np.linalg.norm(xa)
    nrm = np.cross(p2 - p1, p3 - p1)
    if np.dot(nrm, nrm) < 1e-8:
        return np.zeros(3)
    nrm = nrm / np.linalg.norm(nrm)
    ya = np.cross(nrm, xa); ya = ya / np.linalg.norm(ya)
    X2, Y2 = np.dot(p2 - origin, xa), np.dot(p2 - origin, ya)
    X3, Y3 = np.dot(p3 - origin, xa), np.dot(p3 - origin, ya)
    X4, Y4 = np.dot(p4 - origin, xa), np.dot(p4 - origin, ya)
    det = X2*X2*(X3-X4) + X3*X3*(X4-X2) + X4*X4*(X2-X3)
    if abs(det) < 1e-10:
        return np.zeros(3)
    a = (Y2*(X3-X4) + Y3*(X4-X2) + Y4*(X2-X3)) / det
    b = (X2*X2*(Y3-Y4) + X3*X3*(Y4-Y2) + X4*X4*(Y2-Y3)) / det
    c = (X2*X2*(X3*Y4-X4*Y3) + X3*X3*(X4*Y2-X2*Y4) + X4*X4*(X2*Y3-X3*Y2)) / det
    if abs(a) < 1e-8:
        return np.zeros(3)
    xv = -b / (2 * a)
    yv = c - b * b / (4 * a)
    return origin + xv * xa + yv * ya


def vss1(v3, v, r2, num=100):
    """返回 points1, points2 (各 num 点) + 诊断量"""
    v_flat = np.asarray(v3, F)
    v_2 = np.asarray(v, F)
    R_ray = q_from_to([0, 0, 1.0], v_flat)
    R_ray_inv = np.array([R_ray[0], -R_ray[1], -R_ray[2], -R_ray[3]])
    v_bent = q_rot(R_ray_inv, v_2)
    v_bent = v_bent / np.linalg.norm(v_bent)
    R_space = q_from_to(v_flat, v_bent)
    R_total = q_mul(R_ray, R_space)
    axis = np.cross(v_flat, v_bent)
    deltaAngle = np.degrees(np.arccos(np.clip(np.dot(v_flat/np.linalg.norm(v_flat),
                                              v_bent/np.linalg.norm(v_bent)), -1, 1)))
    R = q_axis_angle(np.radians(deltaAngle), axis)
    ez_bent = q_rot(R, [0, 0, 1.0])
    zLength = np.sqrt(r2)
    z2 = 1.0 - zLength
    v_final = q_rot(R_space, ez_bent) * zLength
    V_3 = np.cross(v_final, v_2)
    V_3 = V_3 / np.linalg.norm(V_3)
    V_A = V_3 * z2
    V_D = -V_3 * z2
    v_c = v_final - V_A - (v_final / np.linalg.norm(v_final)) * z2
    v_c1 = v_final + V_A
    V_q1 = v_final + v_2 - V_A - (v_final / zLength) * z2
    V_q2 = v_final + v_2 + V_A
    i1, i2 = find_intersections(v_final, V_A, V_3, v_c, v_c1)
    vz2 = get_vertex_from_four_points(i1, i2, V_q1, V_q2)
    p1, _ = get_parabola_segment_points(v_final, V_A, V_D, num)
    p2, _ = get_parabola_segment_points(vz2, V_q1, V_q2, num)
    diag = dict(deltaAngle=deltaAngle, zLength=zLength, z2=z2,
                R_total=R_total, v_final=v_final, vz2=vz2, v_bent=v_bent)
    return p1, p2, diag


# --------------------------------------------- ComputeAllFeatures (sjy1.cs:386)
def compute_all_features(pa, pb):
    N = min(len(pa), len(pb))
    J = pb[:N] - pa[:N]
    d = np.linalg.norm(J, axis=1)
    ds = np.zeros(N)
    for i in range(1, N):
        ds[i] = 0.5 * (np.linalg.norm(pa[i]-pa[i-1]) + np.linalg.norm(pb[i]-pb[i-1]))
    if N > 1:
        ds[0] = ds[1]
    eps = 1e-6
    I_arr = np.zeros(N); V_arr = np.zeros(N)
    for i in range(N):
        if i == 0:
            step = max(ds[1], eps); gJ = (J[1]-J[0])/step
        elif i == N-1:
            step = max(ds[N-1], eps); gJ = (J[N-1]-J[N-2])/step
        else:
            step = max(0.5*(ds[i]+ds[i+1]), eps); gJ = (J[i+1]-J[i-1])/(2*step)
        normGrad = np.linalg.norm(gJ) + eps
        R = max(d[i]/normGrad, eps)
        I_arr[i] = R
        x = float(np.clip(1.0 - (d[i]*d[i])/(4*R*R), 0, 1))
        V_arr[i] = R**4 * betainc(2.5, 0.5, x)
    return I_arr, V_arr


def load_points(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        raw = list(raw.values())
    return np.array(raw, F)


if __name__ == "__main__":
    P = load_points(r"C:\Users\23128\My project (2)\Assets\Resources\points.json")
    print("=" * 76)
    print("数据: points.json  N =", len(P))
    mu = P.mean(axis=0); C = np.cov((P - mu).T)
    ev, evec = np.linalg.eigh(C)
    order = np.argsort(ev)[::-1]
    ev = ev[order]; evec = evec[:, order]
    r = ev / ev.sum()
    print(f"  PCA 方差比 r = {r[0]:.4f}, {r[1]:.4f}, {r[2]:.4f}   -> zLength = sqrt(r[2]) = {np.sqrt(r[2]):.4f}")
    rp = float(np.abs(P).mean())
    print(f"  rp = mean|p| = {rp:.4f}")

    pc1 = evec[:, 0].copy()
    if pc1[1] < 0:
        pc1 = -pc1
    d2 = np.degrees(np.arccos(np.clip(pc1[1], -1, 1)))
    fj = np.degrees(np.arctan2(pc1[2], pc1[0]))
    if fj < 0:
        fj += 360
    v3 = np.array([np.sin(np.radians(d2))*np.cos(np.radians(fj)),
                   np.cos(np.radians(d2)),
                   np.sin(np.radians(d2))*np.sin(np.radians(fj))])
    print(f"  v3 = {v3}   d2 = {d2:.2f} deg   fj = {fj:.2f} deg")

    # ---------------- Q1: points1/points2 是 (v3, v, r2) 的函数吗 ?
    print()
    print("=" * 76)
    print("Q1  points1/points2 的自由度")
    print("=" * 76)
    v0 = np.array([0.31, 0.72, -0.62]); v0 = v0/np.linalg.norm(v0)
    p1, p2, dg = vss1(v3, v0, r[2])
    print(f"  deltaAngle(v_flat,v_bent) = {dg['deltaAngle']:.4f} deg")
    print(f"  zLength = {dg['zLength']:.6f}   z2 = {dg['z2']:.6f}")
    print(f"  |points1| = {len(p1)}, |points2| = {len(p2)}")
    print(f"  points1 张成的奇异值 = {np.linalg.svd(p1 - p1.mean(0), compute_uv=False)}")
    print(f"  points2 张成的奇异值 = {np.linalg.svd(p2 - p2.mean(0), compute_uv=False)}")

    print("\n  扰动输入, 看 V2 变化 (单次重算, 相对变化):")
    base_I, base_V = compute_all_features(p1, p2)
    base = np.column_stack([base_I, base_V])
    print(f"    baseline:  I in [{base_I.min():.4e}, {base_I.max():.4e}]  "
          f"V in [{base_V.min():.4e}, {base_V.max():.4e}]")
    for name, dv3, dv, dr2 in [("v3 转 0.5deg", 1, 0, 0), ("v 扰动 1%", 0, 1, 0), ("r[2] 扰动 1%", 0, 0, 1)]:
        if dv3:
            ax = np.cross(v3, [0, 0, 1.0]); ax /= np.linalg.norm(ax)
            v3p = q_rot(q_axis_angle(np.radians(0.5), ax), v3)
        else:
            v3p = v3
        vp = v0 + (rng.normal(0, 0.01, 3) if dv else 0)
        vp = vp/np.linalg.norm(vp)
        r2p = r[2]*(1.01 if dr2 else 1.0)
        q1, q2, _ = vss1(v3p, vp, r2p)
        I1, V1 = compute_all_features(q1, q2)
        cur = np.column_stack([I1, V1])
        rel = np.linalg.norm(cur - base)/max(np.linalg.norm(base), 1e-30)
        print(f"    {name:16s}  ||d(I,V)||/||(I,V)|| = {rel:.6f}")

    # ---------------- Q2: V2 在自己 2 维空间里是 1 维吗 ?
    print()
    print("=" * 76)
    print("Q2  V2 = (I/maxI, V/maxV) 的内部维度")
    print("=" * 76)
    V2 = np.column_stack([base_I/base_I.max(), base_V/base_V.max()])
    li, lv = np.log(base_I), np.log(base_V)
    A = np.column_stack([li, np.ones_like(li)])
    sol, *_ = np.linalg.lstsq(A, lv, rcond=None)
    pred = A @ sol
    ss_res = ((lv-pred)**2).sum(); ss_tot = ((lv-lv.mean())**2).sum()
    print(f"  log V = {sol[0]:.4f} * log I + {sol[1]:.4f}    R^2 = {1-ss_res/ss_tot:.6f}")
    cov = np.cov(V2.T)
    ev2 = np.sort(np.linalg.eigvalsh(cov))[::-1]
    print(f"  V2 点云的 PCA 方差比 = {ev2[0]/ev2.sum():.6f}, {ev2[1]/ev2.sum():.6f}")
    print(f"  corr(I, V) = {np.corrcoef(base_I, base_V)[0,1]:.6f}")
    ph = np.arctan2(V2[:, 1], V2[:, 0])
    ii = np.arange(len(V2))
    print(f"  corr(曲线参数 i, 相位 atan2(V,I)) = {np.corrcoef(ii, ph)[0,1]:+.6f}")
    print(f"  相位 atan2(V,I) 单调 ? {np.all(np.diff(ph) > 0) or np.all(np.diff(ph) < 0)}"
          f"   (range {np.degrees(ph.min()):.2f} .. {np.degrees(ph.max()):.2f} deg)")

    # ---------------- Q3: V2 只有 5 个全局自由度 ?
    print()
    print("=" * 76)
    print("Q3  V2 是几个全局数的像 ?")
    print("=" * 76)
    print("  Vss1 只读入: v_flat = v3 (单位向量, 2 DOF), v_2 = vss (单位向量, 2 DOF),")
    print("               zLength = sqrt(r[2]) (1 DOF)   ->  共 5 DOF")
    print("  点集本身 (200 个 3D 点) 除这 5 个数外没有别的入口。")
    print(f"  实际: V2.shape = {V2.shape} = {V2.size} 个数, 由 5 个数生成。")
    print()
    print("  由 v3 决定的量:")
    print(f"    v_final  = {dg['v_final']}   |v_final| = {np.linalg.norm(dg['v_final']):.6f} = zLength")
    print(f"    vz2      = {dg['vz2']}")
    print("  两条曲线 points1 / points2 分别在过 v_final 和过 vz2 的抛物线上。")
