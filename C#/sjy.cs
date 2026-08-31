using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using Unity.VisualScripting;
using UnityEngine;
using Quaternion = UnityEngine.Quaternion;
using Vector2 = UnityEngine.Vector2;
using Vector3 = UnityEngine.Vector3;
using Vector4 = UnityEngine.Vector4;
public class sjy 
{
    public List<Vector3> vector3s;
    public float d2;          // jh 计算出的半顶角
    public float fj;                // 方位角（度）
    public double[] r = new double[3]; // PCA 方差比
    public float rp;                // 母线长
    public Vector3 v3;              // 主轴方向
    public int a2;
    public float e3;// 点云总数（在 jh 
    public UnityEngine.Vector3 vss;
    public List<Vector3> uvss;
    public Vector2[] uvs;
    int gridH = 50;
    private int gridPhi = 36;
    private float sigma = 0.5f;    // 高斯涂抹带宽
    private float[,] probabilityMap;
    private float minU, maxU, minV, maxV;
    public float jh(int a1)
    {
        a2 = vector3s.Count;
        if (vector3s.Count <= 0f) return float.NaN;
        float e = Mathf.Pow(vector3s.Count, 1f / a1);

        float d1 = (float)Math.Asin(e / (1 + e * e));
        d2 = (float)(45 - (d1 * 180 / Math.PI));
        double d3 = (Mathf.Asin(e) - d1) * 180 / Math.PI;
        if (!double.IsNaN(d3) && d2 <= 30.0f) fjh(d1); else { fsy(d1, e); }
        ;
        Debug.Log(d2);
        return d2;

    }
    public void GaussianSplat(List<(Vector2 uv, float weight)> samples)
    {
        // 清空
        for (int i = 0; i < gridH; i++)
            for (int j = 0; j < gridPhi; j++)
                probabilityMap[i, j] = 0f;

        // 边界
        float minU = float.MaxValue, maxU = float.MinValue;
        float minV = float.MaxValue, maxV = float.MinValue;

        foreach (var s in samples)
        {
            if (s.uv.x < minU) minU = s.uv.x;
            if (s.uv.x > maxU) maxU = s.uv.x;
            if (s.uv.y < minV) minV = s.uv.y;
            if (s.uv.y > maxV) maxV = s.uv.y;
        }
        float pad = 3 * sigma;
        minU -= pad; maxU += pad;
        minV -= pad; maxV += pad;
        this.minU = minU; this.maxU = maxU;
        this.minV = minV; this.maxV = maxV;

        float stepU = (maxU - minU) / (gridH - 1);
        float stepV = (maxV - minV) / (gridPhi - 1);
        float sigma2 = 2 * sigma * sigma;
        float cutoff = 9 * sigma * sigma;

        for (int i = 0; i < gridH; i++)
        {
            float u = minU + i * stepU;
            for (int j = 0; j < gridPhi; j++)
            {
                float v = minV + j * stepV;
                float sum = 0f;
                foreach (var s in samples)
                {
                    float du = u - s.uv.x;
                    float dv = v - s.uv.y;
                    float dist2 = du * du + dv * dv;
                    if (dist2 > cutoff) continue;
                    sum += s.weight * Mathf.Exp(-dist2 / sigma2);
                }
                probabilityMap[i, j] = sum;
            }
        }

    }

    public void fjh(float d1)

    {
        double k = Math.Sin(d1);
        double dk = 1 - 4 * k * k;
        double jk = Math.Sqrt(dk);
        double e1 = (1 - jk) / (2 * k);
        double e2 = (1 + jk) / (2 * k);
        float sj = (float)((e2 * e2 - 1f) / (e2 * e2 + 1f));
        float jd2 = Mathf.Asin(sj) * Mathf.Rad2Deg;
    }
    public void fsy(float d1, float e)
    {
        Complex d0 = Complex.Asin(e);
        Complex d3 = (d0 - d1) * 180 / Math.PI;//复数角
        double k = Math.Sin(d2 * Mathf.Deg2Rad);
        double dk = 1 - 4 * k * k;
        double jk = Math.Sqrt(dk);
        Complex complexDk = new Complex(dk, 0);
        Complex jk_complex = Complex.Sqrt(complexDk);
    }
    public float pca()
    {
        int dim = 3;
        var matrix = DenseMatrix.Create(vector3s.Count, dim, (i, j) =>
        {
            Vector3 p = vector3s[i];
            if (j == 0) return p.x;
            if (j == 1) return p.y;
            return p.z;
        });
        var pca = new PCAScikitLearn();
        pca.Fit(matrix, nComponents: 3);
        r[0] = pca.ExplainedVarianceRatio[0];
        r[1] = pca.ExplainedVarianceRatio[1];
        r[2] = pca.ExplainedVarianceRatio[2];
        Vector3 pc1 = new Vector3(
    (float)pca.Components[0, 0],
    (float)pca.Components[0, 1],
    (float)pca.Components[0, 2]);
        float absX = Mathf.Abs(pc1.x);
        float absY = Mathf.Abs(pc1.y);
        float absZ = Mathf.Abs(pc1.z);
        float max = Mathf.Max(absX, absY, absZ);
        float fj = Mathf.Atan2(pc1.y, pc1.x) * Mathf.Rad2Deg;

        return fj;
    }
   
    public Vector3[][] dn()
    {
        int jj = (int)(360 / d2);
        List<float> s = new List<float>();
        for (int i = 0; i < jj; i++)
        {
            float angleDeg = i * d2;
            if (angleDeg == 0 || angleDeg == 90)
                continue;
            float v = (float)Math.Tan(angleDeg * Mathf.Deg2Rad);
            s.Add(v);
        }

        Vector3[][] vector3s1 = new Vector3[jj][];
        for (int i = 0; i < jj; i++)
        {
            vector3s1[i] = new Vector3[a2];
        }

        for (int i2 = 0; i2 < a2; i2++)
        {
            for (int i = 0; i < s.Count; i++)
            {
                Vector3 v = new Vector3(
                    (float)(vector3s[i2].x * s[i] * r[0]),
                    (float)(vector3s[i2].y * s[i] * r[1]),
                    (float)(vector3s[i2].z * s[i] * r[2])
                );
                vector3s1[i][i2] = v;
            }
        }
        return vector3s1;
    }
    public Vector3[][] point2;
    public Vector3[][] sx(Vector3[][] vector3s1)
    {
        // 1. 计算主轴方向 v3
        v3 = new Vector3(
            Mathf.Sin(d2 * Mathf.Deg2Rad) * Mathf.Cos(fj * Mathf.Deg2Rad),
            Mathf.Cos(d2 * Mathf.Deg2Rad),
            Mathf.Sin(d2 * Mathf.Deg2Rad) * Mathf.Sin(fj * Mathf.Deg2Rad)
        );
        Debug.Log(v3.normalized);
        if (uvss == null) uvss = new List<Vector3>();
        if (!uvss.Contains(v3)) uvss.Add(v3);
        e3 = (float)(Math.Cos(d2 * Mathf.Deg2Rad) * 2 / (1 + Math.Sin(d2 * Mathf.Deg2Rad)));



        // 2. 生成所有扇区的方向向量 vcsArray
        Vector3 cs = Vector3.Cross(v3, Vector3.up).normalized;
        int jj = (int)(360 / d2);
        Vector3[] vcsArray = new Vector3[jj];
        for (int i = 0; i < jj; i++)
            vcsArray[i] = Quaternion.AngleAxis(i * d2, v3) * cs;

        // 3. 计算总点数
        int totalPoints = 0;
        for (int s = 0; s < vector3s1.Length; s++)
            totalPoints += vector3s1[s].Length;

        // 4. 结果数组
        Vector3[][] vectors2 = new Vector3[2][];
        vectors2[0] = new Vector3[totalPoints];
        vectors2[1] = new Vector3[totalPoints];

        // 5. 计算每个扇区的局部 prmax（原始最大垂直距离）
        float[] prmax = new float[jj];
        for (int sectorIdx = 0; sectorIdx < jj; sectorIdx++)
        {
            Vector3[] points = vector3s1[sectorIdx];
            if (points == null || points.Length == 0) continue;
            Vector3 vcs = vcsArray[sectorIdx];
            float maxPerp = 0f;
            foreach (Vector3 point in points)
            {
                Vector3 Ap = point - vcs;
                Vector3 perp = Ap - Vector3.Project(Ap, vcs);
                float mag = perp.magnitude;
                if (mag > maxPerp) maxPerp = mag;
            }
            prmax[sectorIdx] = maxPerp;
        }

        // ★ 计算全局最大半径（所有扇区中最大的 prmax） ★
        rp = 0;
        for (int i = 0; i < jj; i++)
            if (prmax[i] > rp) rp = prmax[i];
        // 若所有 prmax 均为 0，则设为 0（或极小值以防除零，但这里仅输出）

        // 6. 第二遍：用原始 prmax 生成圆上交点（保持实际尺度）
        int idx = 0;
        for (int sectorIdx = 0; sectorIdx < jj; sectorIdx++)
        {
            Vector3[] points = vector3s1[sectorIdx];
            if (points == null || points.Length == 0) continue;
            Vector3 vcs = vcsArray[sectorIdx];
            float prmax_local = prmax[sectorIdx];

            foreach (Vector3 point in points)
            {
                Vector3 Ap = point - vcs;
                Vector3 pro = Vector3.Project(Ap, vcs);
                Vector3 perp = Ap - pro;
                float perpMag = perp.magnitude;
                float a = (float)Math.Sqrt(prmax_local * prmax_local - perpMag * perpMag);
                Vector3 dir = perpMag > 1e-6f ? perp.normalized : Vector3.zero;
                Vector3 v1 = pro + dir * a;
                Vector3 v2 = pro - dir * a;
                vectors2[0][idx] = v1;
                vectors2[1][idx] = v2;
                idx++;
            }
        }
        this.point2 = vectors2;
        return vectors2;
    }

    public float[] ComputeProbabilities(List<Vector3> points, float d2, Vector3 v31)
    {
        float d2Rad = d2 * Mathf.Deg2Rad;
        float h = rp * Mathf.Cos(d2Rad);
        float r = rp * Mathf.Sin(d2Rad);
        float a = (rp + r) * 0.5f;
        Vector3 u1 = Vector3.Cross(v31, Vector3.up).normalized;
        if (u1.sqrMagnitude < 1e-6f) u1 = Vector3.Cross(v31, Vector3.forward).normalized;
        Vector3 center = h * v31;
        float c = h * e3;

        Vector3 f1 = c * v31;
        Vector3 f2 = -c * v31;
        float[] fff = new float[points.Count];
        BatchProbability(points, f1, f2, a, fff);
        return fff;
    }

    public static void BatchProbability(List<Vector3> points, Vector3 F1, Vector3 F2, float a, float[] fff)
    {
       
        for (int i = 0; i < points.Count; i++)
        {
            Vector3 p = points[i];
            float d1 = Vector3.Distance(p, F1);
            float d2 = Vector3.Distance(p, F2);
            float delta = d1 + d2 - 2f * a;
            fff[i] = Mathf.Exp(-Mathf.Pow(delta, 2) /8f * a * a);
        }

    }
    public Vector2[] MapDoubleConeToLeaf(List<Vector3> points)
    {
        // 新锥几何参数
        Vector3[] v1;
        float alpha = d2 * Mathf.Deg2Rad;
        float h = rp * Mathf.Cos(alpha);             // 原锥高
        float R = rp * Mathf.Sin(alpha);
        // 原底面半径
        float H = Mathf.PI*R;
        float theta = Mathf.Atan2(0.5f * h, H);// 新锥底面半径
        var sols = ComplexAngleSolver.FindSolutions(alpha);
        float theta2 = 0f;
        //缩小a
        foreach (var z in sols)
        {

            if (Mathf.Abs((float)z.Imaginary) < 1e-6f)   // 虚部足够小，视为实数
            {
                theta2 = (float)z.Real;
                break;
            }
        }
        if (theta2 != 0)
        {
            v1 = th2(theta2, H);

        }
        else
        {
            v1 = th2(theta, H);
        }

        Debug.Log((v1[0] - v1[1]).normalized);
        uvs = new Vector2[points.Count];
        uvss = new List<Vector3>();
        for (int i = 0; i < points.Count; i++)
        {
            Vector3[] v;
            if (theta2 != 0)
            {
                v = th2(theta2, points[i].y);

            }
            else
            {
                v = th2(theta, points[i].y);
            }
           
            float h1 = Vector3.Distance(v[0], v[1]);
            float y = h1 / h;

            float Vc = (float)(y * y * y * 1 / 3 * Math.PI * H * 1 / 4 * h);
            Vector3 u2 = (v[0] - v[1]).normalized;
            Quaternion q = Quaternion.FromToRotation(v3, u2);
            Vector3 C = new Vector3(0, y * H, 0);



            float c2 = (q * points[i]).magnitude;
            float[] f = dd2(Vector3.zero, C, q * points[i]);
            float c3 = f[0];


            float x = Mathf.Acos(1f / Mathf.Sqrt(c3 / c2));

            float x2 = Mathf.Acos(1f / Mathf.Sqrt(c3 / h1));
            float a12 = x * (Mathf.PI / 2f) * Mathf.Rad2Deg;
            float b1 = x2 * (Mathf.PI / 2f) * Mathf.Rad2Deg;
            float vabc = (float)(4 / 3 * Math.PI * f[1] * f[2] * 1 / 2 * h1);


            Vector3 vector = new Vector3(f[1] * Mathf.Sin(a12) * Mathf.Cos(b1), f[2] * Mathf.Sin(a12) * Mathf.Cos(b1), 1 / 2 * h1 * Mathf.Cos(a12)).normalized;


            Quaternion rot = Quaternion.FromToRotation(vector, q * points[i]);

            float phi = Quaternion.Angle(rot, Quaternion.identity);


            float opi = (float)Math.Pow(vabc / Vc, 2);

            if (opi > 1000)
            {
                opi = y * 1000;
            }
            uvs[i] = new Vector2(opi * Mathf.Cos(phi), opi * Mathf.Sin(phi));
            th4(uvs[i]);
        }
      
        return uvs;
    }
    public Vector3 th4(Vector2 uvs)
    {
        float Rs = Mathf.Sqrt(uvs.x * uvs.x + uvs.y * uvs.y);
        float thetas = Mathf.Atan2(uvs.y, uvs.x);
        if (thetas < 0) thetas += 2 * Mathf.PI;
        float r = rp * Mathf.Pow(Rs, (float)Mathf.Sin(d2 * Mathf.Deg2Rad));
        float phis = thetas * (float)Mathf.Sin(d2 * Mathf.Deg2Rad);
        float rs = r * Mathf.Sin(d2 * Mathf.Deg2Rad);
        float zs = r * Mathf.Cos(d2 * Mathf.Deg2Rad);
        Quaternion qs = Quaternion.FromToRotation(Vector3.forward, v3.normalized);
        Quaternion qs2 = Quaternion.FromToRotation(Vector3.forward, -v3.normalized);
        Vector3 vs3 = new Vector3(rs * Mathf.Cos(phis), rs * Mathf.Sin(phis), zs);
        uvss.Add((qs * vs3).normalized);
        uvss.Add((qs2 * vs3).normalized);
        return vs3;
    }
    public Complex InverseTh4(Vector3 vs3,float d2)
    {
        d2 = d2*Mathf.Deg2Rad;
        // 1. 计算 rs, zs, phis
        float rs = Mathf.Sqrt(vs3.x * vs3.x + vs3.y * vs3.y);
        float zs = vs3.z;
        float phis = Mathf.Atan2(vs3.y, vs3.x);
        if (phis < 0) phis += 2 * Mathf.PI;

        // 2. 计算 r 和 thetas
        float r = Mathf.Sqrt(rs * rs + zs * zs);
        float sin_d2 = Mathf.Sin(d2 );
        float cos_d2 = Mathf.Cos(d2 );
        // 验证：rs = r * sin_d2, zs = r * cos_d2，若不一致则可能有误差
        // 但假设数据一致，直接由 phis 求 thetas
        float thetas = phis / sin_d2;
        // 将 thetas 归一化到 [0, 2π)
        while (thetas < 0) thetas += 2 * Mathf.PI;
        while (thetas >= 2 * Mathf.PI) thetas -= 2 * Mathf.PI;

        // 3. 计算 Rs
        float Rs = Mathf.Pow(r / rp, 1.0f / sin_d2);

        // 4. 构造 uvs
        float u = Rs * Mathf.Cos(thetas);
        float v = Rs * Mathf.Sin(thetas);
     
        return new Complex(u, v);
    }
    public Vector3[] th2(float theta, float H)
    {
        float sinRot = Mathf.Sin(theta);
        float cosRot = Mathf.Cos(theta);
        float tanX = -H / sinRot;
        // 上半段 (原始 tan > 0)
        // 因为 Y' = -tan(x) * sin(rotX)
        float x = Mathf.Atan(tanX);
        Vector3 c2 = new Vector3(
            x,
            H,
            tanX * cosRot
        );

        // 下半段 (原始 tan < 0)
        float tanX_lower = H / sinRot;        // Y' = tan(x) * sin(rotX)
        float x_lower = Mathf.Atan(tanX_lower);
        Vector3 c1 = new Vector3(
            x_lower,
            H,
            tanX_lower * cosRot
        );
        return new Vector3[] { c1, c2 };
    }
    public float[] dd2(Vector3 A, Vector3 B, Vector3 C)
    {
        // 1. 计算平面法向量
        Vector3 n = Vector3.Cross(B - A, C - A);  // 以 A 为参考点更安全




        n.Normalize();
        Vector3 e1 = (C - A).normalized;          // 以 AC 方向为 X 轴
        Vector3 e2 = Vector3.Cross(n, e1).normalized;

        // 2. 投影三个点到该平面（以 A 为原点）
        Vector2 A2 = Vector2.zero;
        Vector2 B2 = new Vector2(Vector3.Dot(B - A, e1), Vector3.Dot(B - A, e2));
        Vector2 C2 = new Vector2(Vector3.Dot(C - A, e1), Vector3.Dot(C - A, e2));

        // 3. 重心
        Vector2 center = (A2 + B2 + C2) / 3f;

        // 4. 协方差矩阵（使用投影后的二维坐标！）
        float sxx = (A2.x - center.x) * (A2.x - center.x) +
                    (B2.x - center.x) * (B2.x - center.x) +
                    (C2.x - center.x) * (C2.x - center.x);
        float syy = (A2.y - center.y) * (A2.y - center.y) +
                    (B2.y - center.y) * (B2.y - center.y) +
                    (C2.y - center.y) * (C2.y - center.y);
        float sxy = (A2.x - center.x) * (A2.y - center.y) +
                    (B2.x - center.x) * (B2.y - center.y) +
                    (C2.x - center.x) * (C2.y - center.y);

        // 5. 特征值
        float trace = sxx + syy;
        float det = sxx * syy - sxy * sxy;
        float disc = Mathf.Sqrt(Mathf.Max(0, trace * trace - 4f * det));
        float lambda1 = (trace + disc) * 0.5f;
        float lambda2 = (trace - disc) * 0.5f;

        // 6. 半轴（最大面积内切椭圆的半轴 = sqrt(λ/2)）
        float a_ell = Mathf.Sqrt(Mathf.Max(0, lambda1 / 2f));
        float b_ell = Mathf.Sqrt(Mathf.Max(0, lambda2 / 2f));
        if (b_ell < 1e-6f) b_ell = a_ell * 0.1f;  // 防止退化导致除零

        // 7. 周长（拉马努金近似）
        float C_ell = Mathf.PI * (3f * (a_ell + b_ell) -
                      Mathf.Sqrt((3f * a_ell + b_ell) * (a_ell + 3f * b_ell)));


        // 8. 绕长轴旋转的椭球体积



        return new float[] { C_ell, a_ell, b_ell };
    }
    public static (float A, Vector3 Bvec) FitDispersion(
     Dictionary<Vector3, float[]> Q,
     float[] freqBins)
    {
        List<Vector3> xs = new List<Vector3>();
        List<float> fs = new List<float>();
        int N = freqBins.Length;
        foreach (var kv in Q)
        {
            Vector3 dir = kv.Key;
            float[] arr = kv.Value;
            float sumW = 0f, sumF = 0f;
            for (int i = 0; i < N; i++)
            {
                sumW += arr[i];
                sumF += arr[i] * freqBins[i];
            }
            if (sumW > 1e-8f)
            {
                xs.Add(dir);
                fs.Add(sumF / sumW);
            }
        }
        int n = xs.Count;
        if (n < 4) // 至少需要4个点才能拟合三维平面
        {
            Debug.LogWarning("数据点不足，无法可靠拟合三维色散。");
            return (0f, Vector3.zero);
        }

        // 2. 构造矩阵 A (n x 4) 和向量 b (n)
        // 方程：f = A + Bx*x + By*y + Bz*z
        // 即 [1, x, y, z] * [A, Bx, By, Bz]^T = f
        // 用正规方程求解： (X^T X) * coeff = X^T f
        float[,] XtX = new float[4, 4];
        float[] Xtf = new float[4];

        for (int i = 0; i < n; i++)
        {
            float x = xs[i].x, y = xs[i].y, z = xs[i].z;
            float f = fs[i];
            // 构造行向量 [1, x, y, z]
            float[] row = { 1f, x, y, z };
            // 累加 X^T X
            for (int r = 0; r < 4; r++)
                for (int c = 0; c < 4; c++)
                    XtX[r, c] += row[r] * row[c];
            // 累加 X^T f
            for (int r = 0; r < 4; r++)
                Xtf[r] += row[r] * f;
        }

        // 3. 解线性方程组 (高斯消元)
        float[] coeff = SolveLinearSystem(XtX, Xtf);
        float A = coeff[0];
        Vector3 Bvec = new Vector3(coeff[1], coeff[2], coeff[3]);

        return (A, Bvec);
    }

    // 高斯消元（4x4）
    static float[] SolveLinearSystem(float[,] A, float[] b)
    {
        int n = 4;
        // 增广矩阵
        float[,] aug = new float[n, n + 1];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
                aug[i, j] = A[i, j];
            aug[i, n] = b[i];
        }

        // 前向消元
        for (int i = 0; i < n; i++)
        {
            // 选主元
            int maxRow = i;
            for (int k = i + 1; k < n; k++)
                if (Mathf.Abs(aug[k, i]) > Mathf.Abs(aug[maxRow, i]))
                    maxRow = k;
            // 交换行
            for (int j = i; j <= n; j++)
            {
                float tmp = aug[i, j];
                aug[i, j] = aug[maxRow, j];
                aug[maxRow, j] = tmp;
            }
            if (Mathf.Abs(aug[i, i]) < 1e-12f) continue;

            // 消去下方
            for (int k = i + 1; k < n; k++)
            {
                float factor = aug[k, i] / aug[i, i];
                for (int j = i; j <= n; j++)
                    aug[k, j] -= factor * aug[i, j];
            }
        }

        // 回代
        float[] x = new float[n];
        for (int i = n - 1; i >= 0; i--)
        {
            x[i] = aug[i, n];
            for (int j = i + 1; j < n; j++)
                x[i] -= aug[i, j] * x[j];
            x[i] /= aug[i, i];
        }
        return x;
    }
    public Dictionary<Vector3, float[]> q = new Dictionary<Vector3, float[]>();
    public void xl1(Vector3 v3, List<Vector3> points, float[] vt)
    {
        if (!q.ContainsKey(v3))
        {
            q.Add(v3, vt);
        
        }
       
        for (int i = 0; i < points.Count; i++)
        {

            List<Vector3> y = new List<Vector3>();
            Quaternion q1 = Quaternion.FromToRotation(points[i].normalized, uvss[i]);
            for (int x = 0; x < points.Count; x++)
            {
                y.Add(q1 * points[x]);

            }

            float[] fff = ComputeProbabilities(y, d2, v3);
            if (!q.ContainsKey(uvss[i]))
            {
                q.Add(uvss[i], fff);
            }


        }
        float[] freqBins = new float[points.Count];
        for (int i = 0; i < points.Count; i++)
        {
            freqBins[i] = i;
        }
        var (a22, b22) = FitDispersion(q, freqBins);
        this.vss = b22;
        Debug.Log(vss.normalized);
    }
    
    public List<Vector3> points1;
    public List<Vector3> points2;
    public Quaternion R_total;
    public void Vss1(Vector3 v, out Vector3 V, out Vector3 V_A, out Vector3 V_D,
                 out Vector3 V_q1, out Vector3 V_q2)
    {

        Vector3 v_flat = v3;
       ;
        Vector3 v_2 = v;   // (minDir - maxDir).normalized

        // 求 R_ray 及其逆
        Quaternion R_ray = Quaternion.FromToRotation(Vector3.forward, v_flat);
        Quaternion R_ray_inv = Quaternion.Inverse(R_ray);   // 等价于 FromToRotation(v_flat, Vector3.forward
        // 反解 v_bent
        Vector3 v_bent = (R_ray_inv * v_2).normalized;
        float fj = Mathf.Atan2(v_bent.z, v_bent.x) * Mathf.Rad2Deg;
        float d2_new_rad = Mathf.Acos(v_bent.y);          // 弧度
        float d2_new_deg = d2_new_rad * Mathf.Rad2Deg;    // 度
        if (fj < 0) fj += 360f;
        Quaternion R_space = Quaternion.FromToRotation(v_flat, v_bent);

        // 乘积：R_ray × R_space（空间先，折线后）
         R_total = R_ray * R_space;
        Vector3 axis = Vector3.Cross(v_flat, v_bent).normalized;
        // 旋转角
        float deltaAngle = Vector3.Angle(v_flat, v_bent);
        // 旋转矩阵
        Quaternion R = Quaternion.AngleAxis(deltaAngle, axis);
        Vector3 ex = new Vector3(1, 0, 0);
        Vector3 ey = new Vector3(0, 1, 0);
        Vector3 ez = new Vector3(0, 0, 1);
        Vector3 ex_bent = R * ex;
        Vector3 ey_bent = R * ey;
        Vector3 ez_bent = R * ez;
        float zLength = Mathf.Sqrt((float)r[2]); // 因为 r 是方差，标准差是其平方根，代表实际长度
        float z2 = 1 - zLength;
        Vector3 v_final = R_space * ez_bent * zLength;//圆锥的数据分布
        Vector3 V_3 = Vector3.Cross(v_final, v_2).normalized;

        V_A = Vector3.zero + V_3 * z2;
        V_D = Vector3.zero - V_3 * z2;
        V = v_final;
        Vector3 d = v_final.normalized;
        Vector3 n = V_3.normalized;
        float a = zLength * 0.5f; // 沿光轴方向的延伸
        float b = z2;             // 沿法线方向的弯曲
        Vector3 v_c = v_final - V_A - v_final.normalized * z2;
        Vector3 v_c1 = v_final + V_A;
        V_q1 = v_final + v_2 - V_A - (v_final / zLength) * z2;
        V_q2 = v_final + v_2 + V_A;
        var (inter1, inter2) = FindIntersections(v_final, V_A, V_3, v_c, v_c1);
        Debug.Log((inter1 - inter2).normalized);
        Vector3 vz2 = GetVertexFromFourPoints(inter1, inter2, V_q1, V_q2);
        var (points1, en1) = GetParabolaSegmentPoints(v_final, V_A, V_D);
        var (points2, en2) = GetParabolaSegmentPoints(vz2, V_q1, V_q2);
        this.points1 = points1;
        this.points2 = points2;
    }
   
    public static Vector3 GetVertexFromFourPoints(Vector3 p1, Vector3 p2, Vector3 p3, Vector3 p4)
    {
        // 1. 建立平面局部坐标系
        Vector3 origin = p1;
        Vector3 xAxis = (p2 - p1).normalized;
        if (xAxis.sqrMagnitude < 1e-8f)
        {
            Debug.LogError("p1 和 p2 重合，无法建立坐标系");
            return Vector3.zero;
        }

        // 平面法线（取 p1,p2,p3）
        Vector3 normal = Vector3.Cross(p2 - p1, p3 - p1).normalized;
        if (normal.sqrMagnitude < 1e-8f)
        {
            Debug.LogError("三个点共线，无法确定平面");
            return Vector3.zero;
        }
        Vector3 yAxis = Vector3.Cross(normal, xAxis).normalized;
        float X2 = Vector3.Dot(p2 - origin, xAxis);
        float Y2 = Vector3.Dot(p2 - origin, yAxis);
        float X3 = Vector3.Dot(p3 - origin, xAxis);
        float Y3 = Vector3.Dot(p3 - origin, yAxis);
        float X4 = Vector3.Dot(p4 - origin, xAxis);
        float Y4 = Vector3.Dot(p4 - origin, yAxis);

        // 3. 用三个点 (p2, p3, p4) 求解 y = a x^2 + b x + c
        // 构造线性方程组：
        // [x2^2, x2, 1] [a]   [y2]
        // [x3^2, x3, 1] [b] = [y3]
        // [x4^2, x4, 1] [c]   [y4]
        float x2 = X2, y2 = Y2;
        float x3 = X3, y3 = Y3;
        float x4 = X4, y4 = Y4;

        float det = x2 * x2 * (x3 - x4) + x3 * x3 * (x4 - x2) + x4 * x4 * (x2 - x3);
        if (Mathf.Abs(det) < 1e-10f)
        {
            Debug.LogError("三个点的 x 坐标线性相关，无法求解");
            return Vector3.zero;
        }

        float a = (y2 * (x3 - x4) + y3 * (x4 - x2) + y4 * (x2 - x3)) / det;
        float b = (x2 * x2 * (y3 - y4) + x3 * x3 * (y4 - y2) + x4 * x4 * (y2 - y3)) / det;
        float c = (x2 * x2 * (x3 * y4 - x4 * y3) + x3 * x3 * (x4 * y2 - x2 * y4) + x4 * x4 * (x2 * y3 - x3 * y2)) / det;

        // （可选）验证 p1 是否满足该方程，若偏差微小则忽略

        // 4. 计算顶点局部坐标
        if (Mathf.Abs(a) < 1e-8f)
        {
            Debug.LogWarning("a 接近零，不是抛物线，可能是直线");
            return Vector3.zero;
        }
        float xv = -b / (2f * a);
        float yv = c - b * b / (4f * a);

        // 5. 转回世界坐标
        Vector3 vertex = origin + xv * xAxis + yv * yAxis;
        return vertex;
    }
    public static (Vector3 inter1, Vector3 inter2) FindIntersections(
        Vector3 v_final, Vector3 V_A, Vector3 V_3, Vector3 v_c, Vector3 v_c1)
    {
        float zLength = v_final.magnitude;
        float z2 = V_A.magnitude; // 等于 V_A 的长度
        Vector3 e1 = v_final.normalized;
        Vector3 e2 = V_3.normalized; // V_3 已经是单位向量吗？但确保归一化

        // 将直线端点投影到局部坐标系 (e1, e2)
        float u0 = Vector3.Dot(v_c, e1);
        float v0 = Vector3.Dot(v_c, e2);
        float u1 = Vector3.Dot(v_c1, e1);
        float v1 = Vector3.Dot(v_c1, e2);

        float du = u1 - u0;
        float dv = v1 - v0;

        // 构造二次方程 A*s^2 + B*s + C = 0，s 为直线参数 t ∈ [0,1]
        float A = (zLength / (z2 * z2)) * dv * dv;
        float B = (zLength / (z2 * z2)) * 2 * v0 * dv + du;
        float C = (zLength / (z2 * z2)) * v0 * v0 + u0 - zLength;

        float discriminant = B * B - 4 * A * C;
        List<float> sVals = new List<float>();

        if (discriminant >= 0)
        {
            float sqrtD = Mathf.Sqrt(discriminant);
            float s1 = (-B - sqrtD) / (2 * A);
            float s2 = (-B + sqrtD) / (2 * A);
            if (s1 >= 0 && s1 <= 1) sVals.Add(s1);
            if (s2 >= 0 && s2 <= 1) sVals.Add(s2);
            sVals.Sort();
        }

        if (sVals.Count >= 2)
        {
            Vector3 inter1 = v_c + sVals[0] * (v_c1 - v_c);
            Vector3 inter2 = v_c + sVals[1] * (v_c1 - v_c);

            return (inter1, inter2);
        }
        else
        {
            // 降级方案，返回端点
            return (v_c, v_c1);
        }

    }
    private static float ComputeBendEnergy(float zLength, float z2, Vector3 e2, Vector3 start, Vector3 end, int segments)
    {
        // 获取局部 v 坐标
        float v_start = Vector3.Dot(start, e2);
        float v_end = Vector3.Dot(end, e2);
        if (v_start > v_end) { float tmp = v_start; v_start = v_end; v_end = tmp; }

        float k = (2f * zLength) / (z2 * z2);
        float f(float v) => (k * k) / Mathf.Pow(1f + k * k * v * v, 2.5f);

        // 辛普森积分
        float h = (v_end - v_start) / segments;
        float sum = f(v_start) + f(v_end);
        for (int i = 1; i < segments; i++)
        {
            float v = v_start + i * h;
            sum += (i % 2 == 0) ? 2f * f(v) : 4f * f(v);
        }
        return (h / 3f) * sum;
    }
    public static (List<Vector3> points, float energy) GetParabolaSegmentPoints(
    Vector3 v_final, Vector3 start, Vector3 end, int numPoints = 100)
    {
        float zLength = v_final.magnitude;
        Vector3 e1 = v_final.normalized;

        // 用 start 和 end 确定与 e1 正交的局部坐标轴 e2
        Vector3 delta = end - start;
        // 去掉 delta 在 e1 方向的分量，得到横向分量
        Vector3 perp = delta - Vector3.Dot(delta, e1) * e1;
        if (perp.sqrMagnitude < 1e-9f)
        {
            // 退化：start 和 end 都在对称轴上，无法确定平面，返回直线段
            List<Vector3> linePoints = new List<Vector3>(numPoints);
            for (int i = 0; i < numPoints; i++)
            {
                float t = i / (float)(numPoints - 1);
                linePoints.Add(Vector3.Lerp(start, end, t));
            }
            return (linePoints, 0f);
        }
        Vector3 e2 = perp.normalized;

        // 用 start 和 end 在 e2 上的投影确定积分区间
        float z2 = perp.magnitude * 0.5f;          // 半宽（基于区间长度的一半）
        float v_start = Vector3.Dot(start, e2);
        float v_end = Vector3.Dot(end, e2);

        // 确保 v_start < v_end
        float minV = Mathf.Min(v_start, v_end);
        float maxV = Mathf.Max(v_start, v_end);

        // 采样抛物线点
        List<Vector3> points = new List<Vector3>(numPoints);
        for (int i = 0; i < numPoints; i++)
        {
            float t = i / (float)(numPoints - 1);
            float vCoord = minV + (maxV - minV) * t;
            float uCoord = zLength - (zLength / (z2 * z2)) * vCoord * vCoord;
            Vector3 pt = uCoord * e1 + vCoord * e2;
            points.Add(pt);
        }

        // 计算弯曲能量
        float energy = ComputeBendEnergy(zLength, z2, e2, start, end, segments: numPoints * 2);

        return (points, energy);
    }
    public float[] AverageProbabilitySequence(Dictionary<Vector3, float[]> Q)
    {
        int len = Q.First().Value.Length;
        float[] sum = new float[len];

        foreach (var kv in Q)
        {
            float[] arr = kv.Value;
            for (int i = 0; i < len; i++)
            {
                sum[i] += arr[i];
            }
        }

        // 求平均
        float count = Q.Count;
        for (int i = 0; i < len; i++)
        {
            sum[i] /= count;
        }

        return sum;
    }
    public int com;
    public void rbf(List<Vector3> points1, List<Vector3> points2, float[] vt)
    {
        LocalLensVolumeExtractor.ComputeAllFeatures(points1, points2,
       out float[] I_arr, out float[] V_arr);
        Vector2[] V2 = new Vector2[I_arr.Length];
        float maxI = I_arr.Max(), maxV = V_arr.Max();
        for (int i = 0; i < I_arr.Length; i++)
        {
            V2[i] = new Vector2(I_arr[i] / maxI, V_arr[i] / maxV);
            th4(V2[i]);
        }
        float[] w_float = AverageProbabilitySequence(q);
        double[] w = Array.ConvertAll(w_float, v => (double)v);
        Vector4[] x = MapUVToS3(uvs, V2);
        for (int i = 0; i < x.Length; i++) x[i].Normalize();

        double totalAngle = 0;
        int pairCount = 0;
        for (int i = 0; i < x.Length; i++)
        {
            for (int j = i + 1; j < x.Length; j++)
            {
                float dot = Vector4.Dot(x[i], x[j]);
                dot = Mathf.Clamp(dot, -1f, 1f);
                double angle = Math.Acos(dot);
                totalAngle += angle;
                pairCount++;
            }
        }

        double avgAngle = totalAngle / pairCount;
        double sigma = avgAngle * 0.8;
        double lambda = 1e-3;
        var rbfModel = new SphericalRBF(x, w, sigma, lambda);
        Vector4 testPoint = new Vector4(0.5f, 0.3f, 0.7f, 0.4f);
        testPoint.Normalize();
        double predictedValue = rbfModel.Predict(testPoint);
        
        double pred0 = rbfModel.Predict(x[0]);
      
        double[] rbfWeights = rbfModel.GetWeights();
        float[] externalPotential = new float[uvs.Length];
        for (int i = 0; i < uvs.Length; i++)
        {
            // 计算该流线方向在色散主轴 Bvec 上的投影
            // 这就是该流线受到的"外力势能"
            externalPotential[i] = Vector3.Dot(uvss[i], vss);

        }
        Complex[] uvComplex = uvs.Select(uv => new Complex(uv.x, uv.y)).ToArray();
        Complex[] gradStandard = NumericalGradient(uvs, externalPotential, 12);

        // 第 3 步：转换为复梯度 ∂/∂z = 0.5 * (dw/du - i * dw/dv)
        Complex[] AB_vals = new Complex[uvs.Length];
        for (int i = 0; i < uvs.Length; i++)
        {
            float dw_du = (float)gradStandard[i].Real;
            float dw_dv = (float)gradStandard[i].Imaginary;
            // 这就是您要的 "色散外力势梯度"！
            AB_vals[i] = 0.5f * new Complex(dw_du, -dw_dv);

        }
        // 1. 预测 w 值（如果已训练好 rbfModel）

        Vector4[] X_all = MapUVToS3(uvs, V2);
        for (int i = 0; i < X_all.Length; i++)
            X_all[i].Normalize();

        // 2. 预测所有点的 w
        float[] w_pred = new float[uvs.Length];
        for (int i = 0; i < uvs.Length; i++)
        {
            w_pred[i] = (float)rbfModel.Predict(X_all[i]);
        }

        // 2. 计算一阶梯度
        Complex[] gradW = NumericalGradient(uvs, w_pred, 12);

        // 3. 计算 Hessian 分量
        // 对 gradW 的实部（∂w/∂u）和虚部（∂w/∂v）分别求梯度
        float[] du = gradW.Select(c => (float)c.Real).ToArray();
        float[] dv = gradW.Select(c => (float)c.Imaginary).ToArray();

        Complex[] grad_du = NumericalGradient(uvs, du, 12); // H_uu + i H_uv
        Complex[] grad_dv = NumericalGradient(uvs, dv, 12); // H_uv + i H_vv

        // 4. 构造刚度投影势
        Complex[] stiffnessProj = new Complex[uvs.Length];
        for (int i = 0; i < uvs.Length; i++)
        {
            float u = uvs[i].x, v = uvs[i].y;
            float H_uu = (float)grad_du[i].Real;
            float H_uv = (float)grad_du[i].Imaginary; // 或 grad_dv[i].Real
            float H_vv = (float)grad_dv[i].Imaginary;

            // 投影：0.5 * H * z
            stiffnessProj[i] = 0.5f * new Complex(H_uu * u + H_uv * v, H_uv * u + H_vv * v);

        }

        xl4(point2[0]);
        float[] avg = new float[uvs.Length];
        for (int k = 0; k < uvs.Length; k++)
        {
            float sum = 0f;
            int count1 = 0;
            foreach (var dir in uvss)
            {
                if (p.ContainsKey(dir))
                {
                    sum += Quaternion.Angle(p[dir][k], R_total);
                    count1++;
                }
            }
            avg[k] = sum / count1;
        }
        Complex[] gradSt = NumericalGradient(uvs, avg, 12);

        // 3. 转换为复梯度 ∂/∂z = 0.5 * (∂/∂u - i ∂/∂v)
        Complex[] quatGrad = new Complex[uvs.Length];
        for (int i = 0; i < uvs.Length; i++)
        {
            float dw_du = (float)gradSt[i].Real;
            float dw_dv = (float)gradSt[i].Imaginary;
            quatGrad[i] = 0.5f * new Complex(dw_du, -dw_dv);

        }
        Complex replacement = InverseTh4(v3, d2);

        com = 70;

        Complex[] totalGrad = new Complex[uvs.Length];
        Complex[] f = new Complex[uvs.Length];



        int gradReplacedCount = 0;
        int uvReplacedCount = 0;

        for (int i = 0; i < uvs.Length; i++)
        {
            totalGrad[i] = 6 * (stiffnessProj[i] + AB_vals[i] + quatGrad[i]);
            bool replaced = false;
            if (double.IsNaN(totalGrad[i].Real) || double.IsNaN(totalGrad[i].Imaginary))
            {
                totalGrad[i] = replacement;
                replaced = true;
            }
            else if (totalGrad[i].Magnitude > com)
            {
                totalGrad[i] = replacement;
                replaced = true;
            }
            if (replaced) gradReplacedCount++;
        }

        for (int i = 0; i < uvs.Length; i++)
        {
            f[i] = new Complex(uvs[i].x, uvs[i].y);
            bool replaced = false;
            if (double.IsNaN(f[i].Real) || double.IsNaN(f[i].Imaginary))
            {
                f[i] = replacement;
                replaced = true;
            }
            else if (f[i].Magnitude > com)
            {
                f[i] = replacement;
                replaced = true;
            }
            if (replaced) uvReplacedCount++;
        }

       
        // 拟合 ABCD
        var (A, B, C, D) = FitPolynomial(f, totalGrad);
        Debug.Log((A, B, C, D));
        Complex delta = Discriminant(A, B, C, D);
        Complex[,] omega = ComputeOmega(A, B, C, D);
        double[,] lattice = BuildLatticeFromOmega(omega);
        List<(Complex x, Complex y)> yzqxs = yzqx(f);
        int count = yzqxs.Count;
        double[,] reducedPoints = new double[count, 4];
        for (int i = 0; i < count; i++)
        {
            var (z1, z2) = yzqxs[i];
            double[] vec = new double[] { z1.Real, z1.Imaginary, z2.Real, z2.Imaginary };
            double[] reduced = ReduceToFundamentalDomain(vec, lattice);
            for (int j = 0; j < 4; j++) reducedPoints[i, j] = reduced[j];
        }

       
        


    }

    public static Complex[,] ComputeOmega(Complex A, Complex B, Complex C, Complex D)
{
    // 计算判别式
    Complex disc = B * B - 4.0 * A * C;
    if (disc == Complex.Zero) throw new DivideByZeroException("判别式为零");

    // 计算模参数 τ
    Complex sqrtDisc = Complex.Sqrt(disc);
    Complex ratio = (B + sqrtDisc) / (B - sqrtDisc);
    Complex tau = Complex.Log(ratio) / (2.0 * Math.PI * Complex.ImaginaryOne);

    // 构造 Ω = [[tau, i], [i, tau]]（可自由选择形式，这里取对称形式）
    Complex[,] omega = new Complex[2, 2];
    omega[0, 0] = tau;
    omega[0, 1] = new Complex(0, 1);
    omega[1, 0] = new Complex(0, 1);
    omega[1, 1] = tau;

    
    return omega;
}
public List<(Complex x, Complex y)> yzqx(Complex[] f)
    {
        int count = uvs.Length;
        var yzqxs = new List<(Complex x, Complex y)>();
        Complex[] r45 = new Complex[count];
        Complex[] r30 = new Complex[count];
        Complex[] rd2 = new Complex[count];
        for (int i =0; i< count; i++)
        {
           r45[i] = InverseTh4(point2[0][i], 45);
            r30[i] = InverseTh4(point2[0][i], 30);
            rd2[i] = InverseTh4(point2[0][i], d2);

        }
       
        for (int i = 0; i < count; i++)
        {
            yzqxs.Add((r30[i], r45[i]));
        }

        // 添加配对 (f[i], rd2[i])
        for (int i = 0; i < count; i++)
        {
            yzqxs.Add((f[i], rd2[i]));
        }

        return yzqxs;
    }
    public static double[] ReduceToFundamentalDomain(double[] vector, double[,] lattice)
    {
        // 将 lattice 转为 MathNet 矩阵
        var mat = Matrix<double>.Build.DenseOfArray(lattice);
        var vec = MathNet.Numerics.LinearAlgebra.Vector<double>.Build.Dense(vector);

        // 求逆矩阵（要求矩阵满秩）
        var inv = mat.Inverse();

        // 计算基坐标 t = inv * vec
        var t = inv * vec;

        // 取每个分量的小数部分，确保在 [0,1)
        for (int i = 0; i < t.Count; i++)
        {
            t[i] = t[i] - Math.Floor(t[i]);
            // 处理浮点误差
            if (t[i] >= 1.0) t[i] = 0.0;
            if (t[i] < 0.0) t[i] += 1.0;
        }

        // 返回基坐标（可直接用于均匀性检验）
        return t.ToArray();
    }
    public static double[,] BuildLatticeFromOmega(Complex[,] omega)
    {
        double[,] lattice = new double[4, 4];
        // 前两列为标准基
        lattice[0, 0] = 1; lattice[1, 0] = 0; lattice[2, 0] = 0; lattice[3, 0] = 0;
        lattice[0, 1] = 0; lattice[1, 1] = 1; lattice[2, 1] = 0; lattice[3, 1] = 0;

        // 第三、四列来自 Ω 的列向量的实虚部
        // Ω = [[a, b], [c, d]]
        Complex a = omega[0, 0];
        Complex b = omega[0, 1];
        Complex c = omega[1, 0];
        Complex d = omega[1, 1];

        lattice[0, 2] = a.Real; lattice[1, 2] = c.Real; lattice[2, 2] = a.Imaginary; lattice[3, 2] = c.Imaginary;
        lattice[0, 3] = b.Real; lattice[1, 3] = d.Real; lattice[2, 3] = b.Imaginary; lattice[3, 3] = d.Imaginary;

        return lattice;
    }
    
    private static (Complex A, Complex B, Complex C, Complex D) FitPolynomial(Complex[] f, Complex[] F)
    {
        int N = f.Length;
        var Xmat = Matrix<Complex>.Build.Dense(N, 4);
        var Yvec = MathNet.Numerics.LinearAlgebra.Vector<Complex>.Build.Dense(N);
        for (int i = 0; i < N; i++)
        {
           
            Complex zBar = Complex.Conjugate(f[i]);
            Xmat[i, 0] = f[i];
            Xmat[i, 1] = zBar;
            Xmat[i, 2] = f[i] * f[i];
            Xmat[i, 3] = zBar * zBar;
            Yvec[i] = F[i];
        }
        // 使用 SVD 分解，对病态矩阵更稳定
        var svd = Xmat.Svd();
        var beta = svd.Solve(Yvec);
        return (beta[0], beta[1], beta[2], beta[3]);
    }

    // ---------- 便利函数：判别式 ----------
    public static Complex Discriminant(Complex A, Complex B, Complex C, Complex D)
    {
        Complex num = B * B - 4.0 * A * C;
        Complex den = D * D - 4.0 * B * C;
        return den.Magnitude > 1e-12 ? num / den : new Complex(double.NaN, double.NaN);
    }
    public Dictionary<Vector3, Quaternion[]> p = new Dictionary<Vector3, Quaternion[]>();
    public Dictionary<Vector3,Quaternion[]> xl4(Vector3[] points)
        {
        if (points == null || points.Length == 0) Debug.Log(1);
        if (!uvss.Contains(v3)) uvss.Add(v3);
        for (int i = 0; i < uvss.Count;i++)
        {
            Quaternion[] quaternions = new Quaternion[points.Length];
            for(int x = 0;x< points.Length;x++)
            {
                Quaternion q = Quaternion.FromToRotation(points[x], uvss[i]);
                quaternions[x] = q;
            }
           
            if (!p.ContainsKey(uvss[i]))
            {
                p.Add(uvss[i],quaternions);
            }
        }
        return p;
       

    }
    private static Complex[] NumericalGradient(Vector2[] uv, float[] values, int K)
    {
        int N = uv.Length;
        Complex[] grad = new Complex[N];
        if (N < 3) return grad;

        for (int i = 0; i < N; i++)
        {
            // 计算距离并排序
            var dists = new (int index, float dist)[N];
            for (int j = 0; j < N; j++)
            {
                float dx = uv[j].x - uv[i].x;
                float dy = uv[j].y - uv[i].y;
                dists[j] = (j, Mathf.Sqrt(dx * dx + dy * dy));
            }
            var neighbors = dists.OrderBy(d => d.dist)
                                 .Where(d => d.dist > 1e-8)   // 过滤重复点
                                 .Take(Math.Min(K, N))
                                 .ToArray();

            int M = neighbors.Length;
            if (M < 3) { grad[i] = Complex.Zero; continue; }

            var A = Matrix<double>.Build.Dense(M, 3);
            var b = MathNet.Numerics.LinearAlgebra.Vector<double>.Build.Dense(M);
            for (int k = 0; k < M; k++)
            {
                int idx = neighbors[k].index;
                A[k, 0] = uv[idx].x;
                A[k, 1] = uv[idx].y;
                A[k, 2] = 1.0;
                b[k] = values[idx];
            }

            // 使用 SVD 求解，增强数值稳定性
            var svd = A.Svd();
            var sol = svd.Solve(b);
            grad[i] = new Complex(sol[0], sol[1]);
        }
        return grad;
    }
    public static (float[] theta1, float theta2, float[] theta3) ExtractAngles(Vector2[] uv, Vector2[] V2)
    {
        int N = uv.Length;
        float[] theta1 = new float[N];
        float[] theta3 = new float[N];

        // 计算 uv 角度
        for (int i = 0; i < N; i++)
        {
            theta1[i] = Mathf.Atan2(uv[i].y, uv[i].x);
            if (theta1[i] < 0) theta1[i] += 2 * Mathf.PI;
        }

        // V2 平均角度（圆形平均）
        float sumSin = 0f, sumCos = 0f;
        foreach (var p in V2)
        {
            float phi = Mathf.Atan2(p.y, p.x);
            sumCos += Mathf.Cos(phi);
            sumSin += Mathf.Sin(phi);
        }
        float theta2 = Mathf.Atan2(sumSin, sumCos);
        if (theta2 < 0) theta2 += 2 * Mathf.PI;

        // 计算 V2 平均半径
        float meanR_V2 = 0f;
        foreach (var p in V2) meanR_V2 += p.magnitude;
        meanR_V2 /= V2.Length;

        // 第三个角度：基于径向比
        for (int i = 0; i < N; i++)
        {
            float r_uv = uv[i].magnitude;
            float ratio = Mathf.Clamp(r_uv / (meanR_V2 + 1e-6f), 0.1f, 10f);
            theta3[i] = Mathf.PI / 2f * (ratio - 1f) / (ratio + 1f);
            theta3[i] = Mathf.Clamp(theta3[i], -Mathf.PI / 2f, Mathf.PI / 2f);
        }

        return (theta1, theta2, theta3);
    }

   
    public static Vector4 AnglesToS3(float theta1, float theta2, float theta3)
    {
        float psi = theta3;   // 径向关系
        float theta = theta2; // V2 平均角度
        float phi = theta1;   // uv 角度

        return new Vector4(
            Mathf.Cos(psi) * Mathf.Cos(theta) * Mathf.Cos(phi),
            Mathf.Cos(psi) * Mathf.Cos(theta) * Mathf.Sin(phi),
            Mathf.Cos(psi) * Mathf.Sin(theta),
            Mathf.Sin(psi)
        );
    }

  
    public static Vector4[] MapUVToS3(Vector2[] uv, Vector2[] V2)
    {
        var (theta1, theta2, theta3) = ExtractAngles(uv, V2);
        int N = uv.Length;
        Vector4[] X = new Vector4[N];
        for (int i = 0; i < N; i++)
            X[i] = AnglesToS3(theta1[i], theta2, theta3[i]);
        return X;
    }
}
