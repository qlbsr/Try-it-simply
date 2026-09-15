# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
可解域分析: c=h*e3 多大时 fj 才能从 j (或 tau) 中被分辨?
指标: 同一 d2 下, 16 个 fj 的 j/tau 分散度 (相对标准差)
"""
import json

import numpy as np

from forward_sjy import forward_m1


def main():
    path = r"C:\Users\23128\My project (2)\Assets\Resources\points.json"
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    P = np.array([np.array(p, float) for p in raw], float)
    rp = float(np.mean(np.linalg.norm(P, axis=1)))

    fjs = np.arange(0, 360, 22.5)
    print(f"{'d2':>4s} {'c=h*e3':>8s} {'tau_im_mean':>12s} {'|j|_median':>11s} "
          f"{'j分散(RStd)':>12s} {'tau分散(RStd)':>13s} {'可分?':>6s}")
    for d2d in range(5, 90, 5):
        js = []
        taus = []
        cs = []
        for fjd in fjs:
            r = forward_m1(P, d2d, float(fjd), rp)
            js.append(r["j"])
            taus.append(r["tau"])
            cs.append(r["info"]["c"])
        js = np.array(js)
        taus = np.array(taus)
        c = float(np.mean(cs))
        # 相对分散度: std / |mean|
        j_rstd = float(np.abs(js.std()) / (np.abs(js.mean()) + 1e-300))
        t_rstd = float(np.abs(taus.std()) / (np.abs(taus.mean()) + 1e-300))
        sep = "YES" if t_rstd > 0.02 else ("weak" if t_rstd > 0.005 else "NO")
        print(f"{d2d:4d} {c:8.4f} {taus.imag.mean():12.4f} {np.median(np.abs(js)):11.1f} "
              f"{j_rstd:12.4f} {t_rstd:13.4f} {sep:>6s}")

    print("\n结论: tau 分散度 > 0.02 → fj 可分辨 (逆问题可解)")
    print("      j 分散度远小于 tau 分散度 → 建议回归 tau 而非 j")


if __name__ == "__main__":
    main()
