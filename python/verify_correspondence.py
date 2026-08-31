# -*- coding: utf-8 -*-
"""
自迭代 与 PCA 的对应性验证:
  对每个已收敛(档1)数据集:
  1. PCA环收敛解 (t1f,t2f), θ_PCA
  2. 从 PCA 解出发跑概率自迭代 300 次: 若停在原地(最大角偏差<5°) → 对应成立
     (PCA 解是概率环不动点); 否则不对应
  3. 自迭代全程最大角差: θ_line(0-90) 与 θ_or(0-180, 保留方向) 是否 <90°
  4. 是否自收敛到固定值/范围
"""
import numpy as np

import nsjy_algorithms as m
import verify_probability_correction as vp
import verify_fixedpoint as vf
from verify_corrected import get_pts

t10, t20 = m.compute_taus()

NAMES = ["ball2", "ball3", "gauss0", "gauss1", "gauss3",
         "gauss2", "ball0", "ball6", "ball7", "gauss4", "gauss6", "gauss8",
         "ellip0", "ellipN3"]


def angs_of(hist, axis):
    line = []
    ori = []
    for _, _, _, dn in hist:
        if dn is None:
            line.append(float("nan"))
            ori.append(float("nan"))
            continue
        d = np.array(dn, float)
        # 直线方向 (忽略±)
        dd = d if np.dot(d, axis) >= 0 else -d
        c = float(np.dot(dd, axis))
        line.append(float(np.degrees(np.arccos(np.clip(c, 0.0, 1.0)))))
        # 保留方向 (0-180)
        c2 = float(np.dot(d, axis))
        ori.append(float(np.degrees(np.arccos(np.clip(c2, -1.0, 1.0)))))
    return np.array(line), np.array(ori)


print("=" * 130)
print("档1 数据集: 自迭代(概率环) 与 PCA 的对应性")
print("=" * 130)
hdr = (f"{'name':8s} {'θ_PCA':>6s} | {'自迭代θ0':>7s} {'θend':>6s} {'θ_line最大':>8s} "
       f"{'θ_or最大':>7s} {'最大|Δθ|':>8s} | {'尾100范围':>8s} {'判定':>8s}")
print(hdr)
for name in NAMES:
    pts = get_pts(name)
    r30, r45, a, axis = vp.prepare(pts)
    # 1) PCA 收敛解
    t1f, t2f, F1f, F2f = m.refine_moduli_by_axis(pts, t10, t20, axis,
                                                 max_iter=40, verbose=False)
    d0 = F1f - F2f
    dn0 = d0 / np.linalg.norm(d0)
    th_pca = vp.ang_line(dn0, axis)
    # 2) 从 PCA 解出发的概率自迭代 300 次
    hist = vf.prob_loop(pts, r30, r45, a, t1f, t2f, n_iter=300, every=10)
    line, ori = angs_of(hist, axis)
    tail = line[-10:]
    spread = float(np.nanmax(tail) - np.nanmin(tail))
    maxdev = float(np.nanmax(np.abs(line - line[0])))
    corr = "对应" if maxdev < 5.0 else "不对应"
    conv = "收敛" if spread <= 3.0 else "游走"
    print(f"{name:8s} {th_pca:6.2f} | {line[0]:7.1f} {line[-1]:6.1f} "
          f"{np.nanmax(line):8.1f} {np.nanmax(ori):7.1f} {maxdev:8.1f} | "
          f"{spread:8.1f}  {corr}/{conv}")

print()
print("说明: '对应' = 从PCA解出发自迭代300次, 最大角偏差<5°(PCA解是概率环不动点)")
print("      'θ_or最大' = 保留方向的夹角(0-180), >90° 表示自迭代翻到了反平行半球")
print("      '尾100范围' = 最后10个采样(100次迭代)的角度范围, ≤3° 视为自收敛")
print("DONE")
