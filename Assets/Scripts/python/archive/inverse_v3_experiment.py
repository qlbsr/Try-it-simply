# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
逆问题实验: tau/j → v3 可学性验证
  正向: v3=(d2,fj) → 概率场 → 复平面 → 复梯度 → ABCD → tau → j   (forward_sjy.forward_m1)
  逆向: 网络 tau(2) → v3(3 单位向量), loss = 1 - cos(夹角)
  评估: 按 d2 分桶 (验证"可解域 d2<~65°"的预测)
加速: uvs 只依赖 d2 → 每个 d2 预计算一次
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from forward_sjy import (inverse_th4, v3_from_angles, compute_probabilities,
                         numerical_gradient, fit_polynomial, tau_from_abcd, j_invariant)

RES = r"C:\Users\23128\My project (2)\Assets\Resources\points.json"


def load_points():
    with open(RES, "r", encoding="utf-8") as f:
        raw = json.load(f)
    P = np.array([np.array(p, float) for p in raw], float)
    rp = float(np.mean(np.linalg.norm(P, axis=1)))
    return P, rp


_uv_cache = {}


def uvs_for_d2(P, d2_deg, rp):
    key = round(d2_deg, 6)
    if key not in _uv_cache:
        _uv_cache[key] = np.array([inverse_th4(p, d2_deg, rp) for p in P])
    return _uv_cache[key]


def forward_cached(P, rp, d2_deg, fj_deg):
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
    return dict(v3=v3, tau=tau, j=j, ABCD=(A, B, C, D), c=info["c"])


def gen_dataset(P, rp, n, seed=0, d2_lo=5.0, d2_hi=70.0):
    rng = np.random.default_rng(seed)
    V, T, J, C, D2, FJ = [], [], [], [], [], []
    t0 = time.time()
    for k in range(n):
        d2 = float(rng.uniform(d2_lo, d2_hi))
        fj = float(rng.uniform(0, 360))
        r = forward_cached(P, rp, d2, fj)
        if not np.isfinite(r["tau"].real) or not np.isfinite(r["j"].real):
            continue
        V.append(r["v3"])
        T.append([r["tau"].real, r["tau"].imag])
        J.append([r["j"].real, r["j"].imag])
        C.append(r["c"])
        D2.append(d2)
        FJ.append(fj)
        if (k + 1) % 500 == 0:
            print(f"  gen {k+1}/{n} ({time.time()-t0:.0f}s)", flush=True)
    return (np.array(V), np.array(T), np.array(J), np.array(C),
            np.array(D2), np.array(FJ))


class InvNet(nn.Module):
    def __init__(self, in_dim, hidden=256, depth=3):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers += [nn.Linear(d, 3)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        v = self.net(x)
        return v / (v.norm(dim=-1, keepdim=True) + 1e-9)


def train_eval(Xtr, Vtr, Xte, Vte, D2te, tag="", epochs=3000, lr=2e-3):
    torch.manual_seed(0)
    net = InvNet(Xtr.shape[1])
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
            loss = (1.0 - (pred * Vtr_t[idx]).sum(-1)).mean()   # 1 - cos
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        if (ep + 1) % 500 == 0:
            print(f"  [{tag}] ep{ep+1} loss={tot/n:.6f} "
                  f"avg_ang={math.degrees(math.acos(min(1.0, max(-1.0, 1-tot/n)))):.3f}°", flush=True)
    with torch.no_grad():
        pred = net(Xte_t)
        cos = (pred * Vte_t).sum(-1).clamp(-1, 1)
        ang = torch.rad2deg(torch.acos(cos)).numpy()
    print(f"  [{tag}] 测试集角度误差: mean={ang.mean():.3f}°  median={np.median(ang):.3f}°  "
          f"p90={np.percentile(ang,90):.3f}°  max={ang.max():.3f}°")
    print(f"  [{tag}] 按 d2 分桶:")
    for lo in range(5, 75, 10):
        m = (D2te >= lo) & (D2te < lo + 10)
        if m.sum() > 0:
            print(f"      d2∈[{lo},{lo+10})°: n={m.sum():4d} mean={ang[m].mean():7.3f}° "
                  f"p90={np.percentile(ang[m],90):7.3f}°")
    return net, ang


def main():
    P, rp = load_points()
    n_tr = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    n_te = 400
    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"sjy_dataset_{n_tr}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        Vtr, Ttr, Jtr, Ctr, D2tr = z["Vtr"], z["Ttr"], z["Jtr"], z["Ctr"], z["D2tr"]
        Vte, Tte, Jte, Cte, D2te = z["Vte"], z["Tte"], z["Jte"], z["Cte"], z["D2te"]
        print(f"载入缓存数据集 {cache}")
    else:
        print(f"点集 200 点 rp={rp:.4f}; 生成 {n_tr} 训练 + {n_te} 测试样本 (d2∈[5,70])")
        Vtr, Ttr, Jtr, Ctr, D2tr, _ = gen_dataset(P, rp, n_tr, seed=1)
        Vte, Tte, Jte, Cte, D2te, _ = gen_dataset(P, rp, n_te, seed=99)
        np.savez(cache, Vtr=Vtr, Ttr=Ttr, Jtr=Jtr, Ctr=Ctr, D2tr=D2tr,
                 Vte=Vte, Tte=Tte, Jte=Jte, Cte=Cte, D2te=D2te)
        print(f"已缓存 {cache}")
    print(f"数据: train {len(Vtr)} test {len(Vte)}")

    # 标准化 tau (量级 ~1) 与 j (量级 1e5) → 用 log 尺度
    def prep_tau(T):
        return (T - T.mean(0)) / (T.std(0) + 1e-12)

    def prep_j(J):
        s = np.sign(J)
        return np.concatenate([s * np.log1p(np.abs(J))], axis=1)

    mu_t, sd_t = Ttr.mean(0), Ttr.std(0) + 1e-12
    Ttr_n, Tte_n = (Ttr - mu_t) / sd_t, (Tte - mu_t) / sd_t
    Jtr_n, Jte_n = prep_j(Jtr), prep_j(Jte)
    jm, js = Jtr_n.mean(0), Jtr_n.std(0) + 1e-12
    Jtr_n, Jte_n = (Jtr_n - jm) / js, (Jte_n - jm) / js

    print("\n=== 逆网络 A: tau → v3 ===")
    train_eval(Ttr_n, Vtr, Tte_n, Vte, D2te, tag="tau→v3")
    print("\n=== 逆网络 B: j → v3 ===")
    train_eval(Jtr_n, Vtr, Jte_n, Vte, D2te, tag="j→v3")
    print("\n=== 逆网络 C: (tau,j) → v3 ===")
    train_eval(np.concatenate([Ttr_n, Jtr_n], 1), Vtr,
               np.concatenate([Tte_n, Jte_n], 1), Vte, D2te, tag="(tau,j)→v3")


if __name__ == "__main__":
    main()
