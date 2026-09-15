# -*- coding: utf-8 -*-
# --- repo-root import bootstrap (experiments/ subfolder) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
C# NelderMead.Minimize (n2sjy2.cs 890-1002) 精确 Python 移植
差异 vs scipy: 初始单纯形步长 = 0.05*|x[k]| (x==0 时 0.05), 每维不同!
"""
import numpy as np


def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def nelder_mead_cs(f, x0, lb, ub, max_evals=100, stop_f=16.0, tol=1e-6):
    """最小化 f(x), 返回 (bestX, bestF, evals)。逐行对应 C# NelderMead.Minimize"""
    n = len(x0)
    X = []
    fx = []
    # 初始单纯形: x0 + 0.05*|x0[k]| (x0[k]==0 → 0.05)
    for i in range(n + 1):
        xi = list(x0)
        if i > 0:
            step = 0.05 if xi[i - 1] == 0 else 0.05 * abs(xi[i - 1])
            xi[i - 1] += step
        xi = [clamp(xi[k], lb[k], ub[k]) for k in range(n)]
        X.append(xi)
    evals = 0
    for i in range(n + 1):
        fx.append(f(np.array(X[i])))
        evals += 1
    bestX = list(X[0])
    bestF = fx[0]

    while evals < max_evals:
        # 按 fx 升序排序 (X 同步)
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                if fx[j] < fx[i]:
                    fx[i], fx[j] = fx[j], fx[i]
                    X[i], X[j] = X[j], X[i]
        if fx[0] < bestF:
            bestF = fx[0]
            bestX = list(X[0])
        if bestF <= stop_f:
            break
        # spread (最差点到前 n 个质心)
        spread = 0.0
        for k in range(n):
            mm = 0.0
            for i in range(n):
                mm += X[i][k]
            mm /= n
            spread += (X[n][k] - mm) ** 2
        if spread < tol:
            break
        # 质心 (去掉最差 X[n])
        centroid = [0.0] * n
        for k in range(n):
            mm = 0.0
            for i in range(n):
                mm += X[i][k]
            centroid[k] = mm / n
        # 反射
        xr = [clamp(centroid[k] + 1.0 * (centroid[k] - X[n][k]), lb[k], ub[k]) for k in range(n)]
        fr = f(np.array(xr))
        evals += 1
        if fr < fx[0]:
            # 扩张
            xe = [clamp(centroid[k] + 2.0 * (xr[k] - centroid[k]), lb[k], ub[k]) for k in range(n)]
            fe = f(np.array(xe))
            evals += 1
            if fe < fr:
                X[n] = xe
                fx[n] = fe
            else:
                X[n] = xr
                fx[n] = fr
        elif fr < fx[n - 1]:
            X[n] = xr
            fx[n] = fr
        else:
            # 收缩
            xc = [clamp(centroid[k] + 0.5 * (X[n][k] - centroid[k]), lb[k], ub[k]) for k in range(n)]
            fc = f(np.array(xc))
            evals += 1
            if fc < fx[n]:
                X[n] = xc
                fx[n] = fc
            else:
                # 整体收缩到最优点
                for i in range(1, n + 1):
                    for k in range(n):
                        X[i][k] = clamp(X[0][k] + 0.5 * (X[i][k] - X[0][k]), lb[k], ub[k])
                for i in range(1, n + 1):
                    fx[i] = f(np.array(X[i]))
                    evals += 1
    return np.array(bestX), bestF, evals
