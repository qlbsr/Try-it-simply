# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
诊断: tau/j 对 v3=(d2,fj) 的映射是否连续?
沿 fj 精细扫描 (固定 d2), 检查 tau / ABCD 的相邻跳变
  - 若 tau 有跳变而 ABCD 连续 → 分支切割 (sqrt/log) 是逆问题的真正瓶颈
  - 则逆网络应以 ABCD (连续 8 维) 为中间量, 而不是直接吃 tau/j
"""
import json

import numpy as np

from forward_sjy import forward_m1

RES = r"C:\Users\23128\My project (2)\Assets\Resources\points.json"


def main():
    with open(RES, "r", encoding="utf-8") as f:
        raw = json.load(f)
    P = np.array([np.array(p, float) for p in raw], float)
    rp = float(np.mean(np.linalg.norm(P, axis=1)))

    print(f"{'d2':>4s} {'|d tau| max':>12s} {'|d tau| med':>12s} {'|d ABCD| max':>13s} "
          f"{'|d j| max':>12s} {'|d v3|':>8s} {'tau 跳变@fj':>12s}")
    for d2d in [10, 30, 50, 65]:
        fjs = np.arange(0, 360, 2.0)
        taus, abcds, js, v3s = [], [], [], []
        for fjd in fjs:
            r = forward_m1(P, d2d, float(fjd), rp)
            taus.append(r["tau"])
            abcds.append(np.array(r["ABCD"]))
            js.append(r["j"])
            v3s.append(r["v3"])
        taus = np.array(taus)
        js = np.array(js)
        v3s = np.array(v3s)
        dtau = np.abs(np.diff(taus))
        dj = np.abs(np.diff(js, axis=0))
        dabcd = np.array([np.linalg.norm(abcds[i + 1] - abcds[i]) for i in range(len(abcds) - 1)])
        dv3 = np.array([np.linalg.norm(v3s[i + 1] - v3s[i]) for i in range(len(v3s) - 1)])
        # 归一化 ABCD 跳变 (以中位数为基准)
        db_med = np.median(dabcd) + 1e-30
        imax = int(np.argmax(dtau))
        print(f"{d2d:4d} {dtau.max():12.4f} {np.median(dtau):12.5f} {dabcd.max()/db_med:13.1f}x "
              f"{dj.max():12.3e} {dv3.max():8.4f} {fjs[imax]:12.0f}")
        # 若最大跳变 > 中位数 50 倍 → 判定不连续
        if dtau.max() > 50 * (np.median(dtau) + 1e-30):
            jump_at = fjs[imax]
            print(f"     ⚠ tau 不连续: fj≈{jump_at:.0f}° 处跳变 {dtau[imax]:.3f} "
                  f"(中位 {np.median(dtau):.5f}, 比值 {dtau.max()/max(np.median(dtau),1e-12):.0f}x)")
            print(f"       同一位置 ABCD 跳变 {dabcd[imax]/db_med:.1f}x 中位 → "
                  f"{'ABCD 也跳' if db_med and dbabcd_check(dabcd, imax) else 'ABCD 连续 (仅 tau 跳)'}")

    print("\n判读: |d tau| max / med 远大于 1 → 分支切割跳变; ABCD 若连续 → 以 ABCD 为中间量")


def dbabcd_check(dabcd, imax):
    return dabcd[imax] > 50 * (np.median(dabcd) + 1e-30)


if __name__ == "__main__":
    main()
