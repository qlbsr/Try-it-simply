"""导出 uv 点云给 C# 测试用 (与 verify_v2_replaces_rbf.py 同一批点)"""
import numpy as np

rng = np.random.default_rng(2024)
N = 200
Rs = 0.15 + 1.25 * np.sqrt(rng.uniform(0, 1, N))
th = rng.uniform(0, 2 * np.pi, N)
uvs = np.column_stack([Rs * np.cos(th), Rs * np.sin(th)])
# 与 Python 侧同一个合成概率场
w = np.sin(2.0 * uvs[:, 0]) * np.cos(1.3 * uvs[:, 1]) + 0.4 * uvs[:, 1]

out = r"C:\Users\23128\AppData\Local\Temp\dsh-Z5CgWk\v2_test\uvcloud.txt"
with open(out, "w", encoding="ascii") as f:
    f.write(f"# u v w   N={N}\n")
    for i in range(N):
        f.write(f"{uvs[i,0]:.9g} {uvs[i,1]:.9g} {w[i]:.9g}\n")
print("wrote", out, N)
