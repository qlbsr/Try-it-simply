"""把 points1/points2 导出成 C# 测试用的文本"""
import numpy as np
from verify_v2_provenance import vss1, load_points

P = load_points(r"C:\Users\23128\My project (2)\Assets\Resources\points.json")
mu = P.mean(0); ev, evec = np.linalg.eigh(np.cov((P - mu).T))
o = np.argsort(ev)[::-1]; ev = ev[o]; evec = evec[:, o]
r = ev / ev.sum()
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

out = r"C:\Users\23128\AppData\Local\Temp\dsh-Z5CgWk\v2_test\points_ab.txt"
with open(out, "w", encoding="ascii") as f:
    f.write(f"# points1 then points2, N={len(p1)}\n")
    for p in p1:
        f.write(f"{p[0]:.9g} {p[1]:.9g} {p[2]:.9g}\n")
    for p in p2:
        f.write(f"{p[0]:.9g} {p[1]:.9g} {p[2]:.9g}\n")
print("wrote", out, len(p1), len(p2))
