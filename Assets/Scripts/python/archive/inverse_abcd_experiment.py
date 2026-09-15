# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
验证: 用连续的 ABCD(8 维) 取代 tau/j 作为逆网络输入/中间量
  A. ABCD → v3      (直接, 连续目标)
  B. ABCD → tau_cont(去跳变) 不需要; 改为 ABCD → (d2,fj)
  C. tau_unwrap: 对 tau 做模群归一(取 |Re|<=0.5)后再回归, 看跳变是否消除
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from forward_sjy import v3_from_angles, compute_probabilities, numerical_gradient, \
    fit_polynomial, tau_from_abcd, j_invariant
from inverse_v3_experiment import load_points, uvs_for_d2, InvNet

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sjy_dataset_abcd_2000.npz")


def forward_abcd(P, rp, d2_deg, fj_deg):
    v3 = v3_from_angles(d2_deg, fj_deg)
    prob, info = compute_probabilities(P, v3, rp, d2_deg)
    uvs = uvs_for_d2(P, d2_deg, rp)
    g = numerical_gradient(uvs, prob, 12)
    Fz = 0.5 * (g[:, 0] - 1j * g[:, 1])
    total = 6.0 * Fz
    total = np.where(np.abs(total) > 70.0, 0.0 + 0.0j, total)
    A, B, C, D = fit_polynomial(uvs, total)
    tau, ratio, disc = tau_from_abcd(A, B, C, D)
    j = j_invariant(tau)
    return dict(v3=v3, ABCD=np.array([A, B, C, D]), tau=tau, j=j, c=info["c"])


def gen(P, rp, n, seed, d2_lo=5.0, d2_hi=70.0):
    rng = np.random.default_rng(seed)
    V, AB, T, J, D2, FJ, C = [], [], [], [], [], [], []
    t0 = time.time()
    for k in range(n):
        d2 = float(rng.uniform(d2_lo, d2_hi))
        fj = float(rng.uniform(0, 360))
        r = forward_abcd(P, rp, d2, fj)
        if not np.isfinite(r["tau"].real) or not np.isfinite(r["j"].real):
            continue
        # ABCD 8 维实数; 归一化到单位尺度 (除以自身范数, 保方向)
        ab = r["ABCD"]
        scale = np.abs(ab).max() + 1e-30
        V.append(r["v3"])
        AB.append(np.stack([(ab / scale).real, (ab / scale).imag]).T.reshape(-1))
        T.append([r["tau"].real, r["tau"].imag])
        J.append([r["j"].real, r["j"].imag])
        D2.append(d2)
        FJ.append(fj)
        C.append(r["c"])
        if (k + 1) % 500 == 0:
            print(f"  gen {k+1}/{n} ({time.time()-t0:.0f}s)", flush=True)
    return (np.array(V), np.array(AB), np.array(T), np.array(J),
            np.array(D2), np.array(FJ), np.array(C))


def train_eval(Xtr, Vtr, Xte, Vte, D2te, tag="", epochs=3000, lr=2e-3, hidden=256, depth=3):
    torch.manual_seed(0)
    net = InvNet(Xtr.shape[1], hidden=hidden, depth=depth)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    Vtr_t = torch.tensor(Vtr, dtype=torch.float32)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)
    Vte_t = torch.tensor(Vte, dtype=torch.float32)
    n = len(Xtr_t)
    bs = 256
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            pred = net(Xtr_t[idx])
            loss = (1.0 - (pred * Vtr_t[idx]).sum(-1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        if (ep + 1) % 1000 == 0:
            print(f"  [{tag}] ep{ep+1} avg_ang={math.degrees(math.acos(min(1.0, max(-1.0, 1-tot/n)))):.3f}°", flush=True)
    with torch.no_grad():
        pred = net(Xte_t)
        cos = (pred * Vte_t).sum(-1).clamp(-1, 1)
        ang = torch.rad2deg(torch.acos(cos)).numpy()
    print(f"  [{tag}] 测试: mean={ang.mean():.3f}° median={np.median(ang):.3f}° "
          f"p90={np.percentile(ang,90):.3f}° max={ang.max():.3f}°")
    for lo in range(5, 75, 15):
        m = (D2te >= lo) & (D2te < lo + 15)
        if m.sum() > 0:
            print(f"      d2∈[{lo},{lo+15})°: n={m.sum():4d} mean={ang[m].mean():7.3f}°")
    return net, ang


def main():
    P, rp = load_points()
    n_tr, n_te = 2000, 400
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        Vtr, ABtr, Ttr, Jtr, D2tr, FJtr, Ctr = (z["Vtr"], z["ABtr"], z["Ttr"], z["Jtr"],
                                                z["D2tr"], z["FJtr"], z["Ctr"])
        Vte, ABte, Tte, Jte, D2te, FJte, Cte = (z["Vte"], z["ABte"], z["Tte"], z["Jte"],
                                                z["D2te"], z["FJte"], z["Cte"])
        print(f"载入缓存 {CACHE}")
    else:
        print(f"生成 {n_tr}+{n_te} 样本 (含 ABCD)...")
        Vtr, ABtr, Ttr, Jtr, D2tr, FJtr, Ctr = gen(P, rp, n_tr, 1)
        Vte, ABte, Tte, Jte, D2te, FJte, Cte = gen(P, rp, n_te, 99)
        np.savez(CACHE, Vtr=Vtr, ABtr=ABtr, Ttr=Ttr, Jtr=Jtr, D2tr=D2tr, FJtr=FJtr, Ctr=Ctr,
                 Vte=Vte, ABte=ABte, Tte=Tte, Jte=Jte, D2te=D2te, FJte=FJte, Cte=Cte)
        print(f"已缓存 {CACHE}")

    # 标准化
    def std(Xtr, Xte):
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
        return (Xtr - mu) / sd, (Xte - mu) / sd

    ABtr_n, ABte_n = std(ABtr, ABte)
    Ttr_n, Tte_n = std(Ttr, Tte)

    print("\n=== A. ABCD(8,连续) → v3 ===")
    train_eval(ABtr_n, Vtr, ABte_n, Vte, D2te, tag="ABCD→v3")
    print("\n=== B. (ABCD,tau) → v3 ===")
    train_eval(np.concatenate([ABtr_n, Ttr_n], 1), Vtr,
               np.concatenate([ABte_n, Tte_n], 1), Vte, D2te, tag="(ABCD,tau)→v3")
    print("\n=== C. 对照: tau → v3 (同数据) ===")
    train_eval(Ttr_n, Vtr, Tte_n, Vte, D2te, tag="tau→v3")


if __name__ == "__main__":
    main()
