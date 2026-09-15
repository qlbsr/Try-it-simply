"""
verify_sjysjy_map.py
--------------------
复刻 C#\sjysjy.cs:233 的 MardenEllipse.MapDoubleConeToLeaf, 在 points.json 上看它到底做了什么。

C# 原文结构:
  alpha = d2*Deg2Rad
  h  = rp*cos(alpha)          // 原锥高
  R  = rp*sin(alpha)          // 原底面半径
  H  = pi*R                   // "高"  (常量)
  HR = h/2                    // 声明后从未使用 (死变量)
  fzj = atan2(0.5h, H)        // theta2 找不到实根时的兜底角
  sols = ComplexAngleSolver.FindSolutions(alpha)
  theta2 = 第一个 |Im|<1e-6 的解, 否则 0
  v1 = th2(theta2, H) 或 th2(fzj, H)
  v3 = normalize(v1[0]-v1[1])        <-- 覆盖传入的 v3 (PCA 轴被丢弃)

  for i:
     v   = th2(theta2, points[i].y) 或 th2(fzj, points[i].y)   <-- 第二参数是点自己的 y
     h1  = |v[0]-v[1]|
     y   = h1/h
     Vc  = y^3 * (1/3)*pi*H * (1/4)*h^2
     u2  = normalize(v[0]-v[1])
     q   = FromToRotation(v3, u2)
     v_  = q * (points[i]*y)
     C   = (0, y*H, 0)
     c2  = |v_|
     f   = dd2(0, C, v_)      // [周长, a, b, cx, cy]
     c3  = f[0]
     x   = c3/c2 ; x2 = c3/h1
     w1  = sqrt(x^2-1) ; w2 = sqrt(x2^2-1)      <-- 无 NaN 防护
     cAxis = |h1*w2 - h1*w1|/(2pi)
     vabc  = pi*cAxis^2*c3
     opi   = (c2/sqrt((h1/2)^2+(yH)^2))^2
     phi   = pi*2*vabc/Vc
     uvs[i]= opi*(cos phi, sin phi)
"""
import numpy as np

F = np.float64
I = np.complex128


# ---------------------------------------------- ComplexAngleSolver
def eq(theta, alpha):
    sa, ca = np.sin(alpha), np.cos(alpha)
    st = np.sin(theta)
    ct = np.cos(theta) / np.sin(theta)          # Cot
    atanval = np.arctan(np.pi * sa / st)
    left = 4.0 * atanval * atanval + 4.0 * np.pi ** 2 * sa * sa * ct * ct
    return left - ca * ca


def newton(guess, alpha, maxiter=100, tol=1e-10):
    th = guess
    for _ in range(maxiter):
        f = eq(th, alpha)
        df = (eq(th + 1e-8, alpha) - eq(th - 1e-8, alpha)) / 2e-8
        if abs(df) < 1e-14:
            break
        d = f / df
        th = th - d
        if abs(d) < tol:
            return th
    return th


def find_solutions(alpha):
    guesses = [np.pi / 4 + 0j, np.pi / 2 + 0j, 1j, -1j,
               np.pi / 4 + 1j, np.pi / 4 - 1j, -np.pi / 4 + 0j, -np.pi / 4 + 1j]
    sols = []
    for g in guesses:
        r = newton(g, alpha)
        if any(abs(r - s) < 1e-6 for s in sols):
            continue
        if not (np.isnan(r.real) or np.isnan(r.imag)):
            sols.append(r)
    return sols


# ------------------------------------------------------------- th2
def th2(theta, H):
    s, c = np.sin(theta), np.cos(theta)
    if abs(s) < 1e-300:
        s = 1e-300 if s >= 0 else -1e-300
    tanX = -H / s
    c2v = np.array([np.arctan(tanX), H, tanX * c], F)
    tanXLow = H / s
    c1v = np.array([np.arctan(tanXLow), H, tanXLow * c], F)
    return c1v, c2v


# ------------------------------------------------------------- dd2
def dd2(A, B, C):
    n = np.cross(B - A, C - A); n = n / np.linalg.norm(n)
    e1 = (C - A); e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(n, e1); e2 = e2 / np.linalg.norm(e2)
    A2 = np.array([0.0, 0.0])
    B2 = np.array([np.dot(B - A, e1), np.dot(B - A, e2)])
    C2 = np.array([np.dot(C - A, e1), np.dot(C - A, e2)])
    ctr = (A2 + B2 + C2) / 3.0
    sxx = ((A2[0]-ctr[0])**2 + (B2[0]-ctr[0])**2 + (C2[0]-ctr[0])**2)
    syy = ((A2[1]-ctr[1])**2 + (B2[1]-ctr[1])**2 + (C2[1]-ctr[1])**2)
    sxy = ((A2[0]-ctr[0])*(A2[1]-ctr[1]) + (B2[0]-ctr[0])*(B2[1]-ctr[1])
           + (C2[0]-ctr[0])*(C2[1]-ctr[1]))
    tr = sxx + syy
    det = sxx * syy - sxy * sxy
    disc = np.sqrt(max(0.0, tr * tr - 4 * det))
    l1 = (tr + disc) * 0.5
    l2 = (tr - disc) * 0.5
    a = np.sqrt(max(0.0, l1 / 2.0))
    b = np.sqrt(max(0.0, l2 / 2.0))
    if b < 1e-6:
        b = a * 0.1
    C_ell = np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))
    return np.array([C_ell, a, b, ctr[0], ctr[1]])


# -------------------------------------------------------- quaternion
def q_from_to(a, b):
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    c = np.cross(a, b); d = float(np.dot(a, b))
    if np.linalg.norm(c) < 1e-12:
        if d > 0:
            return np.array([1.0, 0, 0, 0])
        ax = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
        ax = np.cross(a, ax); ax = ax / np.linalg.norm(ax)
        return np.array([0.0, *ax])
    q = np.array([1.0 + d, c[0], c[1], c[2]])
    return q / np.linalg.norm(q)


def q_rot(q, v):
    w, x, y, z = q
    R = np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                  [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                  [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
    return R @ v


# ------------------------------------------------ MapDoubleConeToLeaf
def map_double_cone_to_leaf(points, d2deg, rp, v3_in=None, verbose=True):
    alpha = np.radians(d2deg)
    h = rp * np.cos(alpha)
    R = rp * np.sin(alpha)
    H = np.pi * R
    HR = 0.5 * h
    fzj = np.arctan2(0.5 * h, H)
    sols = find_solutions(alpha)
    theta2 = 0.0
    for z in sols:
        if abs(z.imag) < 1e-6:
            theta2 = float(z.real)
            break

    if verbose:
        print(f"  alpha = {d2deg:.4f} deg = {alpha:.6f} rad")
        print(f"  h = rp*cos a = {h:.6f}   R = rp*sin a = {R:.6f}   H = pi*R = {H:.6f}")
        print(f"  HR = h/2 = {HR:.6f}  <-- 声明后从未使用")
        print(f"  fzj = atan2(h/2, H) = {np.degrees(fzj):.6f} deg")
        print(f"  FindSolutions 共 {len(sols)} 个:")
        for z in sols:
            isreal = "实根" if abs(z.imag) < 1e-6 else "复根"
            resid = abs(eq(z, alpha))
            print(f"     theta = {z.real:+.10f} {z.imag:+.6e} i   |F| = {resid:.3e}   {isreal}")
        print(f"  => theta2 取 = {theta2:.10f} rad = {np.degrees(theta2):.4f} deg"
              f"   {'(走 th2(theta2,·) 分支)' if theta2 != 0 else '(走 fzj 兜底分支)'}")

    th_use = theta2 if theta2 != 0 else fzj
    v1 = th2(th_use, H)
    v3_new = (v1[0] - v1[1]); v3_new = v3_new / np.linalg.norm(v3_new)
    if verbose:
        print(f"  v1[0]-v1[1] = {v1[0]-v1[1]}")
        print(f"  v3_new = {v3_new}     y 分量 = {v3_new[1]:.3e}")
        print(f"  传入的 v3 = {v3_in}  -> 被覆盖丢弃 (轴完全由锥几何决定)")

    N = len(points)
    uvs = np.zeros((N, 2), F)
    diag = dict(h1=np.zeros(N), y=np.zeros(N), opi=np.zeros(N), phi=np.zeros(N),
                c2=np.zeros(N), c3=np.zeros(N), u2=np.zeros((N, 3)),
                Vc=0.0, theta2=theta2, v3_new=v3_new,
                w1=np.zeros(N), w2=np.zeros(N), Vc_arr=np.zeros(N))
    for i in range(N):
        v = th2(th_use, points[i][1])
        h1 = np.linalg.norm(v[0] - v[1])
        y = h1 / h
        Vc = y ** 3 * (1.0 / 3.0) * np.pi * H * (1.0 / 4.0) * h * h
        u2 = (v[0] - v[1]); u2 = u2 / np.linalg.norm(u2)
        q = q_from_to(v3_new, u2)
        v_ = q_rot(q, points[i] * y)
        C = np.array([0.0, y * H, 0.0])
        c2 = np.linalg.norm(v_)
        f = dd2(np.zeros(3), C, v_)
        c3 = f[0]
        x = c3 / c2
        x2 = c3 / h1
        w1 = np.sqrt(x * x - 1)
        w2 = np.sqrt(x2 * x2 - 1)
        cAxis = abs(h1 * w2 - h1 * w1) / (2.0 * np.pi)
        vabc = np.pi * cAxis * cAxis * c3
        opi = (c2 / np.sqrt((h1 * 0.5) ** 2 + (y * H) ** 2)) ** 2
        phi = np.pi * 2.0 * vabc / Vc
        uvs[i] = (opi * np.cos(phi), opi * np.sin(phi))
        diag['h1'][i] = h1; diag['y'][i] = y; diag['opi'][i] = opi
        diag['phi'][i] = phi; diag['c2'][i] = c2; diag['c3'][i] = c3
        diag['u2'][i] = u2; diag['w1'][i] = w1; diag['w2'][i] = w2
        diag['Vc_arr'][i] = Vc
    return uvs, diag


def load_points(path):
    import json
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        raw = list(raw.values())
    return np.array(raw, F)


if __name__ == "__main__":
    P = load_points(r"C:\Users\23128\My project (2)\Assets\Resources\points.json")
    mu = P.mean(0); ev, evec = np.linalg.eigh(np.cov((P - mu).T))
    o = np.argsort(ev)[::-1]; ev = ev[o]; evec = evec[:, o]
    pc1 = evec[:, 0].copy()
    if pc1[1] < 0:
        pc1 = -pc1
    d2 = np.degrees(np.arccos(np.clip(pc1[1], -1, 1)))
    fj = np.degrees(np.arctan2(pc1[2], pc1[0]))
    if fj < 0:
        fj += 360
    v3_pca = np.array([np.sin(np.radians(d2))*np.cos(np.radians(fj)),
                       np.cos(np.radians(d2)),
                       np.sin(np.radians(d2))*np.sin(np.radians(fj))])
    rp = float(np.abs(P).mean())
    print("=" * 78)
    print(f"points.json  N={len(P)}  d2={d2:.4f} deg  rp={rp:.6f}")
    print("=" * 78)
    uvs, dg = map_double_cone_to_leaf(P, d2, rp, v3_in=v3_pca)

    print()
    print("=" * 78)
    print("逐点量的统计 (看哪些量其实与 i 无关)")
    print("=" * 78)
    for k in ['h1', 'y', 'Vc_arr', 'w1', 'w2', 'c2', 'c3', 'opi', 'phi']:
        a = dg[k]
        print(f"  {k:8s} min {np.nanmin(a):+.6e}  max {np.nanmax(a):+.6e}  "
              f"std {np.nanstd(a):.3e}  唯一值 {len(np.unique(np.round(a,12)))}")
    u2u = dg['u2']
    print(f"  u2 唯一方向数 = {len(np.unique(np.round(u2u,12), axis=0))}"
          f"   首行 {u2u[0]}")

    print()
    print("=" * 78)
    print("uvs 的信息内容")
    print("=" * 78)
    R_uv = np.linalg.norm(uvs, axis=1)
    ang = np.arctan2(uvs[:, 1], uvs[:, 0])
    rp_i = np.linalg.norm(P, axis=1)
    yang = np.arccos(np.clip(P[:, 1] / np.maximum(rp_i, 1e-30), -1, 1))
    print(f"  |uvs| : min {np.nanmin(R_uv):.6e} max {np.nanmax(R_uv):.6e}")
    print(f"  corr(log|uvs|, log|p|)  = {np.corrcoef(np.log(R_uv), np.log(rp_i))[0,1]:+.6f}")
    print(f"  corr(log|uvs|, log opi) = {np.corrcoef(np.log(R_uv), np.log(np.abs(dg['opi'])))[0,1]:+.6f}")
    print(f"  corr(phi,      log|p|)  = {np.corrcoef(dg['phi'], np.log(rp_i))[0,1]:+.6f}")
    print(f"  corr(phi,      angle_to_y) = {np.corrcoef(dg['phi'], yang)[0,1]:+.6f}")
    print(f"  corr(opi, |p|^2)        = {np.corrcoef(dg['opi'], rp_i**2)[0,1]:+.6f}")
    print(f"  opi / |p|^2 的 min/max  = {np.nanmin(dg['opi']/rp_i**2):.6e} / "
          f"{np.nanmax(dg['opi']/rp_i**2):.6e}  (若为常数则 opi 纯属 |p|^2)")

    print()
    print("=" * 78)
    print("闭式验证: |uvs| 是不是恰好 = 4|p|^2/(h^2+4H^2) ?")
    print("=" * 78)
    alpha = np.radians(d2)
    h = rp * np.cos(alpha)
    H = np.pi * rp * np.sin(alpha)
    pred = 4.0 * rp_i ** 2 / (h * h + 4.0 * H * H)
    print(f"  h^2+4H^2 = {h*h+4*H*H:.8f}   4/(h^2+4H^2) = {4.0/(h*h+4*H*H):.10f}")
    print(f"  |uvs| / |p|^2 唯一值 = {np.unique(np.round(dg['opi']/rp_i**2, 10))}")
    print(f"  |uvs| 与 4|p|^2/(h^2+4H^2) 最大相对差 = "
          f"{np.max(np.abs(R_uv - pred)/pred):.3e}")
    print("  推导: opi = (c2/sqrt((h1/2)^2+(yH)^2))^2, c2 = y|p|, y = h1/h")
    print("        => opi = |p|^2 * (h1^2/h^2) / (h1^2/4 + h1^2 H^2/h^2) = 4|p|^2/(h^2+4H^2)")
    print("  即: 叶平面的半径只携带 |p|^2 一个数, 与方向、与锥几何都无关。")

    print()
    print("=" * 78)
    print("反事实: 若 FindSolutions 的猜测顺序不同, 取到 theta2 = pi/2 会怎样 ?")
    print("=" * 78)
    th_pi2 = np.pi / 2
    h1s = np.array([np.linalg.norm(th2(th_pi2, P[i][1])[0] - th2(th_pi2, P[i][1])[1])
                    for i in range(len(P))])
    u2s = np.array([(lambda d: d / np.linalg.norm(d))(
        th2(th_pi2, P[i][1])[0] - th2(th_pi2, P[i][1])[1]) for i in range(len(P))])
    pred_h1 = 2.0 * np.abs(np.arctan(P[:, 1]))
    print(f"  theta2 = pi/2 时:  h1 = 2|atan(points[i].y)| ?  最大相对差 = "
          f"{np.max(np.abs(h1s - pred_h1)/np.maximum(np.abs(pred_h1),1e-30)):.3e}")
    print(f"                    h1 唯一值数 = {len(np.unique(np.round(h1s,10)))}  (仍逐点变化)")
    print(f"                    u2 唯一方向数 = {len(np.unique(np.round(u2s,10), axis=0))}"
          f"   取值: {np.unique(np.round(u2s,6), axis=0).tolist()}")
    print(f"                    v3_new = {dg['v3_new']} (theta2=-8.84 分支)")
    print("  => theta2 = pi/2 时 c1-c2 = (2*atan(Y), 0, 2Y*cos(pi/2)/1), cos(pi/2)~6e-17,")
    print("     所以 z 分量被抹掉, u2 只剩 +-(1,0,0) —— 即'洞口对齐 x 轴'的强制规范化。")
    print("     q = FromToRotation(v3_new, +-e_x) 只有 2 个取值, 逐点的旋转结构被毁掉;")
    print("     而 h1 = 2*atan(points[i].y) 仍然逐点变化, 只是不再含方向信息。")
    print("     (本次实测取到 theta2 = -8.84, 所以走的是另一支 —— 但分支只由猜测顺序决定。)")

    print()
    print("=" * 78)
    print("NaN / Inf")
    print("=" * 78)
    nNaN = int(np.sum(~np.isfinite(uvs)))
    nNaN_w = int(np.sum(~np.isfinite(dg['w1']))) + int(np.sum(~np.isfinite(dg['w2'])))
    print(f"  uvs 里非有限分量 = {nNaN} / {uvs.size}")
    print(f"  w1/w2 非有限 = {nNaN_w} / {2*len(P)}")
    nneg1 = int(np.sum((dg['c3']/dg['c2'])**2 - 1 < 0))
    nneg2 = int(np.sum((dg['c3']/dg['h1'])**2 - 1 < 0))
    print(f"  x^2-1 < 0 的点数 = {nneg1}   x2^2-1 < 0 的点数 = {nneg2}   (这些点 w 会是 NaN)")
    print(f"  points[i].y == 0 的点数 = {int(np.sum(P[:,1] == 0))}   "
          f"(h1=0 -> y=0 -> Vc=0 -> phi 除零)")
