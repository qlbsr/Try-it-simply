"""d 对 vss 的依赖: 多组 vss 扫一遍"""
import sys, io
sys.path.insert(0, "experiments")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
from verify_v2_provenance import (load_points, q_from_to, q_rot, find_intersections,
                                  get_vertex_from_four_points, get_parabola_segment_points)

F = np.float64


def vss1_full(v3, v, r2, num=100):
    v_flat = np.asarray(v3, F); v_2 = np.asarray(v, F)
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
    v_c = v_final - V_A - vfh * z2
    v_c1 = v_final + V_A
    V_q1 = v_final + v_2 - V_A - (v_final / zLength) * z2
    V_q2 = v_final + v_2 + V_A
    i1, i2 = find_intersections(v_final, V_A, V_3, v_c, v_c1)
    vz2 = get_vertex_from_four_points(i1, i2, V_q1, V_q2)
    p1, _ = get_parabola_segment_points(v_final, V_A, V_D, num)
    p2, _ = get_parabola_segment_points(v_final, V_q1, V_q2, num)
    return dict(v_final=v_final, V_3=V_3, V_A=V_A, V_D=V_D, v_c=v_c, v_c1=v_c1,
                V_q1=V_q1, V_q2=V_q2, zLength=zLength, z2=z2, vz2=vz2, p1=p1, p2=p2)


P = load_points(r"C:\Users\23128\My project (2)\Assets\Resources\points.json")
mu = P.mean(0); ev, evec = np.linalg.eigh(np.cov((P - mu).T))
o = np.argsort(ev)[::-1]; ev = ev[o]; evec = evec[:, o]
r = ev / ev.sum(); rp = float(np.abs(P).mean())
pc1 = evec[:, 0].copy()
if pc1[1] < 0: pc1 = -pc1
d2 = np.degrees(np.arccos(np.clip(pc1[1], -1, 1)))
fj = np.degrees(np.arctan2(pc1[2], pc1[0]))
if fj < 0: fj += 360
v3 = np.array([np.sin(np.radians(d2))*np.cos(np.radians(fj)), np.cos(np.radians(d2)),
               np.sin(np.radians(d2))*np.sin(np.radians(fj))])

rng = np.random.default_rng(77)
print(f"{'#':>2} {'vss':>36} {'|V3|':>7} {'|vz2|':>10} {'|ch1|':>10} {'|ch2|':>10} {'d med':>10} {'d max':>10}  |d-d0|max")
base = None
for k in range(5):
    vss = rng.normal(size=3); vss /= np.linalg.norm(vss)
    S = vss1_full(v3, vss, r[2])
    d = np.linalg.norm(S['p2'] - S['p1'], axis=1)
    if base is None: base = d
    print(f"{k:>2} [{vss[0]:+.4f},{vss[1]:+.4f},{vss[2]:+.4f}] "
          f"{np.linalg.norm(S['V_3']):>7.4f} {np.linalg.norm(S['vz2']):>10.6f} "
          f"{np.linalg.norm(S['V_D']-S['V_A']):>10.6f} {np.linalg.norm(S['V_q2']-S['V_q1']):>10.6f} "
          f"{np.median(d):>10.6f} {d.max():>10.6f}  {np.abs(d-base).max():.2e}")

S = vss1_full(v3, rng.normal(size=3), r[2])
print()
print("解析:")
print(f"  |ch1| = |V_D-V_A| = 2 z2              = {2*S['z2']:.6f}   <- 与 vss 无关")
print(f"  |ch2| = z2*|2 V_3 + vfh| = z2*sqrt(5) = {S['z2']*np.sqrt(5):.6f}   <- 与 vss 无关 (V_3 _|_ vfh)")
print(f"  |vz2| 与 V_3 的取值随 vss 变化 -> d 却不变: 说明 d 的配对把 vss 抵消掉了")
