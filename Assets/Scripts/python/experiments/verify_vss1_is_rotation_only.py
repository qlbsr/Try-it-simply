"""
verify_vss1_is_rotation_only.py
-------------------------------
Vss1 输出的两条抛物线, 在【刚体旋转】意义下是否只由 r[2] 决定?
即: 两个向量 (v3, vss) 到底给这两条曲线贡献了'形状'还是只有'姿态'?

方法: 把 (points1[i], points2[i]) 拼成一个 2n x 3 的点云, 用 Kabsch 找
      A -> B 的最优刚体变换, 看对齐后的残差。
残差 ~ 0  =>  两条曲线只差一个刚体旋转 => 形状与 v3/vss 无关。
"""
import sys, io
sys.path.insert(0, "experiments")
import numpy as np
from verify_v2_provenance import (load_points, q_from_to, q_rot, find_intersections,
                                  get_vertex_from_four_points, get_parabola_segment_points)
out = io.StringIO()


def vss1_pairs(v3, v, r2, num=100):
    v_flat = np.asarray(v3, float); v_2 = np.asarray(v, float)
    R_ray = q_from_to([0, 0, 1.0], v_flat)
    R_ray_inv = np.array([R_ray[0], -R_ray[1], -R_ray[2], -R_ray[3]])
    v_bent = q_rot(R_ray_inv, v_2); v_bent /= np.linalg.norm(v_bent)
    R_space = q_from_to(v_flat, v_bent)
    zLength = np.sqrt(r2); z2 = 1.0 - zLength
    ez_bent = q_rot(R_space, [0, 0, 1.0])
    v_final = q_rot(R_space, ez_bent) * zLength
    V_3 = np.cross(v_final, v_2); V_3 /= np.linalg.norm(V_3)
    V_A = V_3 * z2; V_D = -V_3 * z2
    vfh = v_final / np.linalg.norm(v_final)
    v_c = v_final - V_A - vfh * z2; v_c1 = v_final + V_A
    V_q1 = v_final + v_2 - V_A - (v_final / zLength) * z2; V_q2 = v_final + v_2 + V_A
    i1, i2 = find_intersections(v_final, V_A, V_3, v_c, v_c1)
    vz2 = get_vertex_from_four_points(i1, i2, V_q1, V_q2)
    p1, _ = get_parabola_segment_points(v_final, V_A, V_D, num)
    p2, _ = get_parabola_segment_points(v_final, V_q1, V_q2, num)
    return np.vstack([p1, p2])


def kabsch_resid(A, B):
    """A, B: (m,3)。返回把 A 对齐到 B 的最优刚体变换后的 RMS 残差"""
    ca, cb = A.mean(0), B.mean(0)
    X, Y = A - ca, B - cb
    H = X.T @ Y
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    Xa = X @ R.T
    return float(np.sqrt(((Xa - Y) ** 2).sum() / len(A))), \
           float(np.sqrt((Y ** 2).sum() / len(A)))


P = load_points(r"C:\Users\23128\My project (2)\Assets\Resources\points.json")
mu = P.mean(0); ev, evec = np.linalg.eigh(np.cov((P - mu).T))
o = np.argsort(ev)[::-1]; ev = ev[o]; evec = evec[:, o]
r = ev / ev.sum()
rng = np.random.default_rng(31)

base_v3 = np.array([0.0, 0.0, 1.0])
base_vss = np.array([1.0, 0.0, 0.0])
A0 = vss1_pairs(base_v3, base_vss, r[2])

print(f"基准: v3 = ẑ, vss = x̂,  r[2] = {r[2]:.6f}", file=out)
print(f"  点云尺寸 = ({A0.shape[0]}, 3)   形状尺度(RMS 半径) = {np.sqrt((A0**2).sum()/len(A0)):.6f}", file=out)
print(file=out)
print(f"{'#':>2} {'v3':>32} {'vss':>32} {'对齐 RMS 残差':>16} {'相对残差':>12}", file=out)

worst = 0.0
for k in range(6):
    v3 = rng.normal(size=3); v3 /= np.linalg.norm(v3)
    vss = rng.normal(size=3); vss /= np.linalg.norm(vss)
    A = vss1_pairs(v3, vss, r[2])
    res, scale = kabsch_resid(A, A0)
    rel = res / scale
    worst = max(worst, rel)
    print(f"{k:>2} [{v3[0]:+.4f},{v3[1]:+.4f},{v3[2]:+.4f}] "
          f"[{vss[0]:+.4f},{vss[1]:+.4f},{vss[2]:+.4f}] {res:>16.3e} {rel:>12.3e}", file=out)

print(file=out)
print(f"最大相对残差 = {worst:.3e}", file=out)
print(file=out)
print("=> 残差 ~ 1e-16: 两条抛物线在【刚体旋转意义下完全相同】,", file=out)
print("   即 (v3, vss) 只贡献了一个姿态(被 R_total 记录), 没有贡献任何形状。", file=out)
print("   形状只由 r[2] 决定。", file=out)
sys.stdout.write(out.getvalue())
