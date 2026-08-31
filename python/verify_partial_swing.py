# -*- coding: utf-8 -*-
"""
部分旋转摆动测试: 参考 rA=d_i(方形格), rB=d0(初始), 每轮向 d_new 旋转 α·angle
  α=1: 全额旋转 (自证预言, |Δ|平凡0)
  α=0.5/0.3: 参考摆动, 等分线移动, 方向追逐
观察: 等分线是否稳定? 最终方向相对 固定原始参考(d_i,d0) 的真 |Δ| 是否≤10?
"""
import math
import time

import numpy as np

import data_driven_axis as dd
import n2sjy2 as n2


def angle(u, v):
    return math.degrees(math.acos(np.clip(np.dot(u, v), -1, 1)))


def setup(pts):
    rp = n2.compute_rp(pts)
    d2 = math.asin(math.cos(math.radians(30)) / math.pi)
    a = rp * (1 + math.sin(d2))
    _, r45, r30, _ = n2.yzqx(pts, rp)
    v = n2.pca_axis(pts)
    t1t, t2t = n2.compute_taus()
    pt0, p10, p20, sig0 = dd.fast_probs(r30, r45, t1t, t2t, a)
    F10, F20 = n2.extract_foci(pts, p10, sig0[1], p20, sig0[2])
    d0 = np.array(F10, float) - np.array(F20, float)
    d0 = d0 / np.linalg.norm(d0)
    pti, p1i, p2i, sigi = dd.fast_probs(r30, r45, complex(0, 1), complex(0, 1), a)
    F1i, F2i = n2.extract_foci(pts, p1i, sigi[1], p2i, sigi[2])
    d_i = np.array(F1i, float) - np.array(F2i, float)
    d_i = d_i / np.linalg.norm(d_i)
    return r30, r45, a, v, d0, d_i


def rotate_toward(ref, target, alpha):
    """把 ref 向 target 旋转 alpha·angle(ref,target)"""
    ref = ref / np.linalg.norm(ref)
    target = target / np.linalg.norm(target)
    axis = np.cross(ref, target)
    nrm = np.linalg.norm(axis)
    if nrm < 1e-12:
        return ref
    axis = axis / nrm
    th = alpha * math.acos(np.clip(np.dot(ref, target), -1, 1))
    return ref * math.cos(th) + np.cross(axis, ref) * math.sin(th) \
        + axis * np.dot(axis, ref) * (1 - math.cos(th))


def partial_swing(pts, alpha, rounds=30, n_inner=15):
    r30, r45, a, v, d0, d_i = setup(pts)
    t1 = t2 = complex(0, 1)
    rA = d_i.copy()
    rB = d0.copy()
    history = []
    for k in range(rounds):
        bis = rA + rB
        bis = bis / np.linalg.norm(bis)
        t1b, t2b, F1, F2 = dd.inner_refine(t1, t2, pts, r30, r45, a, bis, n_inner)
        pt, p1, p2, sig = dd.fast_probs(r30, r45, t1b, t2b, a)
        F1f, F2f = n2.extract_foci(pts, p1, sig[1], p2, sig[2])
        d_new = np.array(F1f, float) - np.array(F2f, float)
        d_new = d_new / np.linalg.norm(d_new)
        # 参考向 d_new 部分旋转 (摆动)
        rA = rotate_toward(rA, d_new, alpha)
        rB = rotate_toward(rB, d_new, alpha)
        # 真 |Δ|: 相对 固定原始参考 (d_i, d0)
        true_delta = abs(angle(d_new, d0) - angle(d_new, d_i))
        history.append((bis.copy(), d_new.copy(), true_delta))
        t1, t2 = t1b, t2b
    return history


def main():
    datasets = {"cube0": dd.make_cube(10), "ellip0": dd.make_ellip(0)}
    for alpha in (1.0, 0.5, 0.3):
        print(f"=== 部分旋转 α={alpha} ===", flush=True)
        for name, pts in datasets.items():
            t0 = time.time()
            hist = partial_swing(pts, alpha, rounds=30)
            r30, r45, a, v, d0, d_i = setup(pts)
            true_d = np.array([h[2] for h in hist])
            final_d = hist[-1][1]
            bis_final = hist[-1][0]
            print(f"  {name}: 真|Δ|(固定参考) min={true_d.min():6.2f}° max={true_d.max():6.2f}° "
                  f"∠(终d,v)={angle(final_d, v):6.1f}° ∠(终d,等分线)={angle(final_d, bis_final):6.1f}° "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        print(flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
