"""
verify_abcd_coreset.py — ABCD vs 点集: 最小点集数量 与 结构选择
================================================================================

用户的优化目标:
    ABCD 是记忆点集的稳定摘要。
    (1) 改变记忆【结构】使 ABCD 的变量减少 —— 点集数量不变, 点集改变
    (2) 同时改变数量与结构使 ABCD 【不变】—— 找到最小点集数量
    (3) 少量记忆也能知道 ABCD 结构

做法:
    memory(3D 点) -> 复平面 z -> 势 U(z) = sum_k exp(-|z-z_k|^2/(2 sigma^2))
                   -> 场 F = dU/dz -> 最小二乘拟合 -> ABCD

    【基已修正】: 用 5 列 [z, zbar, z^2, z zbar, zbar^2]
      原因(verify_abcd_meaning.py 已验证): 4 列基缺 z zbar = |z|^2,
      三次势下拟合残差达 0.13~0.40, 而二次势下只有 1.8e-16。
      4 列基会把 z zbar 的贡献漏进 A/B/C/D, 使它们被污染。

    子集选择对比:
      随机子集          基线
      greedy 残差子集   Leja 型: 每次加入当前拟合残差最大的点 (D-optimal 的廉价近似)
    指标: 相对全集的 ABCD 偏差; 以及同数量下不同子集的 ABCD 离散度

运行: python verify_abcd_coreset.py
"""
import io, json, math, os, sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
PTS = os.path.join(HERE, "..", "..", "..", "Resources", "points.json")
SIGMA = 0.08
BASIS5 = 5
TOL_LIST = [1e-1, 1e-2, 1e-3]


def _flatten(o):
    """兼容多种 JSON 结构: dict 列表 / 数组列表 / 扁平数字 / 含包装键。"""
    if isinstance(o, dict):
        for k in ("points", "Points", "items", "data", "list"):
            if k in o:
                return _flatten(o[k])
        if "x" in o:
            return [[float(o.get("x", 0.0)), float(o.get("y", 0.0)), float(o.get("z", 0.0))]]
        out = []
        for v in o.values():
            out.extend(_flatten(v))
        return out
    if isinstance(o, list):
        if not o:
            return []
        if isinstance(o[0], dict):
            return [[float(d.get("x", 0.0)), float(d.get("y", 0.0)), float(d.get("z", 0.0))]
                    for d in o]
        if isinstance(o[0], (list, tuple)):
            out = []
            for r in o:
                r = [float(v) for v in r]
                while len(r) < 3:
                    r.append(0.0)
                out.append(r[:3])
            return out
        if isinstance(o[0], (int, float)):
            f = [float(v) for v in o]
            if len(f) % 3 == 0:
                return [f[i:i + 3] for i in range(0, len(f), 3)]
            if len(f) % 2 == 0:
                return [[f[i], f[i + 1], 0.0] for i in range(0, len(f), 2)]
        out = []
        for v in o:
            out.extend(_flatten(v))
        return out
    return []


def load_pts():
    for p in (PTS,
              r"C:\Users\23128\My project (2)\Assets\Resources\points.json"):
        if os.path.exists(p):
            raw = json.load(io.open(p, encoding="utf-8"))
            return np.array(_flatten(raw), dtype=float), p
    raise SystemExit("找不到 points.json")


def field(z, zk, sigma=SIGMA):
    """F(z) = dU/dz,  U = sum_k exp(-|z-z_k|^2/(2 s^2))
       dU/dz = sum_k -(z - z_k)/(2 s^2) exp(...)"""
    d = z[:, None] - zk[None, :]
    e = np.exp(-(np.abs(d) ** 2) / (2.0 * sigma * sigma))
    return np.sum(-d / (2.0 * sigma * sigma) * e, axis=1)


def design(z, ncol=BASIS5):
    cols = [z, np.conj(z), z ** 2, z * np.conj(z), np.conj(z) ** 2]
    return np.column_stack(cols[:ncol])


def fit(z, zk, ncol=BASIS5):
    M = design(z, ncol)
    F = field(z, zk)
    coef, _, _, _ = np.linalg.lstsq(M, F, rcond=None)
    coef = list(coef) + [0j] * (4 - len(coef)) if ncol == 4 else list(coef)
    # 统一成 [A, B, C, E, D]（4 列时 E = 0）
    if ncol == 4:
        return np.array([coef[0], coef[1], coef[2], 0j, coef[3]])
    return np.array(coef)


def greedy_choose(z, zk, m, ncol=BASIS5):
    """Leja 型贪心: 每次选当前拟合残差最大的点。"""
    idx = [int(np.argmax(np.abs(z)))]
    while len(idx) < m:
        c = fit(np.array([z[i] for i in idx]), zk, ncol)
        M = design(z, ncol)
        r = np.abs(M @ c - field(z, zk))
        r[idx] = -1.0
        idx.append(int(np.argmax(r)))
    return idx


def part0_basis_check(z, zk):
    print("=" * 78)
    print("0. 基的修正是否重要 (真实数据)")
    print("=" * 78)
    full4 = fit(z, zk, 4)
    full5 = fit(z, zk, 5)
    M4 = design(z, 4)
    M5 = design(z, 5)
    F = field(z, zk)
    r4 = float(np.max(np.abs(M4 @ np.array([full4[0], full4[1], full4[2], full4[4]]) - F)))
    r5 = float(np.max(np.abs(M5 @ full5 - F)))
    print(f"  全集 N = {len(z)} 点")
    print(f"  4 列基 [z,zbar,z^2,zbar^2]        最大残差 = {r4:.6e}")
    print(f"  5 列基 [+ z zbar]                 最大残差 = {r5:.6e}")
    print(f"  残差下降 {r4/r5 if r5>0 else float('inf'):.1f} 倍")
    print()
    print("  通道对比 (4 列 vs 5 列, 看被污染的量级):")
    names = ["A", "B", "C", "E=zzbar", "D"]
    print(f"  {'通道':>9} {'4 列基':>22} {'5 列基':>22} {'差':>12}")
    for i, nm in enumerate(names):
        v4 = full4[i]
        v5 = full5[i]
        print(f"  {nm:>9} {v4.real:>10.6f}{v4.imag:>+11.6f}i "
              f"{v5.real:>10.6f}{v5.imag:>+11.6f}i {abs(v5-v4):>12.3e}")
    print()
    return full5


def part1_structure(z, zk, full):
    print("=" * 78)
    print("1. 数量固定, 改变【结构】能否降低 ABCD 的离散度")
    print("=" * 78)
    rng = np.random.default_rng(20250901)
    for m in (10, 20, 40):
        ntrials = 60 if m <= 20 else 30
        coefs_r = np.array([fit(z[rng.choice(len(z), m, replace=False)], zk)
                            for _ in range(ntrials)])
        dev_r = np.abs(coords_dev(coefs_r, full))
        # greedy 只给一条轨迹, 用不同种子(取 z 的极值点作种子)得多个样本
        coefs_g = []
        for s in range(min(ntrials, 20)):
            zz = np.roll(z, s * max(1, len(z) // 20))
            idx = greedy_choose(zz, zk, m)
            coefs_g.append(fit(z[idx], zk))
        coefs_g = np.array(coefs_g)
        dev_g = np.abs(coords_dev(coefs_g, full))
        print(f"  m = {m:>3}  (相对全集的通道偏差 max|.|)")
        print(f"     随机子集: 最大偏差 {dev_r.max():.4e}   中位 {np.median(dev_r):.4e}"
              f"   离散度(std) {dev_r.std():.4e}")
        print(f"     greedy  : 最大偏差 {dev_g.max():.4e}   中位 {np.median(dev_g):.4e}"
              f"   离散度(std) {dev_g.std():.4e}")
        ratio = dev_r.max() / max(dev_g.max(), 1e-300)
        print(f"     => 同数量下 greedy 把最大偏差压低 {ratio:.1f} 倍")
    print()
    print("  注: greedy 的种子换法有限, 所以 greedy 的样本数少于随机; 结论看最大偏差。")
    print()


def coords_dev(coefs, full):
    """各通道相对全集的偏差(取绝对值最大)。"""
    return np.abs(coefs - full[None, :]).max(axis=1)


def part2_min_count(z, zk, full):
    print("=" * 78)
    print("2. 最小点集: 保持 ABCD 在容差内所需的最少点数")
    print("=" * 78)
    print("  greedy 递增选点, 记录相对全集的 ABCD 偏差")
    print()
    hdr = f"  {'m':>6} {'max|dABCD|':>14} {'|dA|':>11} {'|dB|':>11} " \
          f"{'|dC|':>11} {'|dE|':>11} {'|dD|':>11}"
    print(hdr)
    idx = [int(np.argmax(np.abs(z)))]
    ms, devs = [], []
    for target in (2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256):
        idx = greedy_choose(z, zk, target)
        c = fit(z[idx], zk)
        d = np.abs(c - full)
        ms.append(target); devs.append(d.max())
        print(f"  {target:>6} {d.max():>14.4e} {d[0]:>11.2e} {d[1]:>11.2e} "
              f"{d[2]:>11.2e} {d[3]:>11.2e} {d[4]:>11.2e}")
    print()
    print("  达到各容差所需的最小点数 (greedy 选点):")
    for tol in TOL_LIST:
        need = None
        for m, d in zip(ms, devs):
            if d <= tol:
                need = m
                break
        print(f"     容差 {tol:.0e}  ->  最少 {need if need else '>256'} 点"
              f"  (全集 {len(z)} 点)")
    print()
    return ms, devs


def part3_law(z, zk, full, ms, devs):
    print("=" * 78)
    print("3. 偏差随点数下降的规律 (能否外推)")
    print("=" * 78)
    ms_a = np.array(ms, float)
    dv = np.array(devs, float)
    ok = dv > 0
    lg = np.polyfit(np.log(ms_a[ok]), np.log(dv[ok]), 1)
    print(f"  拟合 log(偏差) = {lg[0]:.4f} * log(m) + {lg[1]:.4f}")
    print(f"  即 偏差 ~ m^({lg[0]:.4f})")
    print()
    print("  读法: 指数接近 -1 表示偏差随点数按 1/m 下降;")
    print("        比 -1 更陡说明 greedy 的结构选择额外加速了收敛。")
    print()
    print(f"  {'m':>6} {'实测偏差':>14} {'幂律外推':>14}")
    for m, d in zip(ms, devs):
        pred = math.exp(lg[1]) * m ** lg[0]
        print(f"  {m:>6} {d:>14.4e} {pred:>14.4e}")


if __name__ == "__main__":
    arr, path = load_pts()
    print(f"点集: {path}   N = {len(arr)}")
    z = arr[:, 0] + 1j * arr[:, 1]
    zk = z.copy()
    print(f"sigma = {SIGMA}")
    print()
    full = part0_basis_check(z, zk)
    part1_structure(z, zk, full)
    ms, devs = part2_min_count(z, zk, full)
    part3_law(z, zk, full, ms, devs)
    print("=" * 78)
    print("结论")
    print("=" * 78)
    print("  1. 4 列基缺 z zbar, 残差远高于 5 列基 -> 通道被污染, 必须先补列。")
    print("  2. 数量固定时, 结构选择(greedy)能把 ABCD 偏差压低若干倍。")
    print("  3. 最小点数由容差决定, 偏差随 m 按幂律下降, 可外推到目标容差。")
    print("  4. 少量记忆即可定 ABCD 结构 —— 上面给出'多少点够'的定量答案。")
