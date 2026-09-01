using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using Unity.VisualScripting;
using UnityEngine;
using static alglib;
using Vector3 = UnityEngine.Vector3;
using Vector4 = UnityEngine.Vector4;

public class nsjy : MonoBehaviour
{
    public Complex[] r45;
    public Complex[] r30;
    public Complex[] r16;
    public Complex[] r74;
    public float rp;
    public float a;
    public List<Vector3> points;
    private void Start()
    {
        var random = new System.Random();
        List<float> fjValues = new List<float>();

        for (int trial = 0; trial < 20; trial++)
        {
            // 1. 生成随机点集（单位球体内）

            for (int i = 0; i < 200; i++)
            {
                Vector3 p = UnityEngine.Random.insideUnitSphere;
                points.Add(p);
            }

        }
        rp = nsjy.ComputeRp(points);
        float d2 = Mathf.Asin(Mathf.Cos(30 * Mathf.Deg2Rad) / Mathf.PI);
        a = rp * (1 + Mathf.Sin(d2));
        float e3 = (float)(Math.Cos(d2) * 2 / (1 + Math.Sin(d2)));
        float h = rp * Mathf.Cos(d2);
        float c = h * e3;



        List<(Complex x, Complex y)> yzqx1 = yzqx(points, out r45,out r30, out r74, rp);
        double Kk = elliptic.ellipticintegralk(0.5f * 0.5f, alglib.xdefault);
        double K_sqrt3_2 = elliptic.ellipticintegralk(Math.Pow((Math.Sqrt(3) / 2.0), 2), alglib.xdefault);
        Complex K_2 = new Complex(0.5 * Kk, -0.5 * K_sqrt3_2);
        double gamma14 = Gamma(0.25);
        Complex K_inv_sqrt2 = gamma14 * gamma14 / (4.0 * Math.Sqrt(Math.PI));
        Complex K_sqrt2 = new Complex(1.0 / Math.Sqrt(2), -1.0 / Math.Sqrt(2)) * K_inv_sqrt2;
        var (t1, t2) = ComputeTaus();
        Debug.Log((t1, t2));
        Complex t11 = new Complex(0, 1);
        Complex t22 = new Complex(0, 1);
        float[][] probs = DeviationCalculator.ComputeProbabilitiesFromTaus(r30, r45, t1, t2,a);
        float[][] probs1 = DeviationCalculator.ComputeProbabilitiesFromTaus(r30, r45, t11, t22, a);

        var (F1, F2) = ExtractFoci(
    points, probs[1], probs[3][1], probs[2], probs[3][2]);
        var (F11, F22) = ExtractFoci(
   points, probs1[1], probs1[3][1], probs1[2], probs1[3][2]);
        Vector3 v31 = (F11 - F22).normalized;
        Vector3 v3 = (F1 - F2).normalized;
        Vector3 v = pca(points);
      
        Vector3 f1 = c* v;
        Vector3 f2 = -c * v;
       // Vector4 v4 = pca4(yzqx1);
      //  Vector4 f41 = c * v4;
       // Vector4 f42 = -c * v4;



        // 计算 K(kTarget)
        double sin16 = Mathf.Sin(16 * Mathf.Deg2Rad);
        Complex K_sin16 = Carlsonfk.K(1 / sin16);

        double sin74 = Mathf.Sin(74 * Mathf.Deg2Rad);
        Complex K_sin74 = Carlsonfk.K(1 / sin74);
        Complex aot = K_sqrt2 - K_2 - K_sin16;
        Complex aot2 = K_sin74 - K_2 - K_sqrt2;
        Complex KF2 = Carlsonfk.NK(aot2);
        Complex KF = Carlsonfk.NK(aot);
        var (t1k, t2k) = n2sjy2.ComputeTaus(KF,KF2);
       

      
       
       


    }
    public static List<(Complex x, Complex y)> yzqx(List<Vector3> points, out Complex[] r45, out Complex[] r30, out Complex[] r74, float rp)
    {
        if (points == null || points.Count == 0)
        {
            r45 = null;
            r30 = null;
            r74 = null;
            return new List<(Complex x, Complex y)>();
        }

        int count = points.Count;
        r45 = new Complex[count];
        r30 = new Complex[count];
        r74 = new Complex[count];
        // 第一步：计算原始值
        for (int i = 0; i < count; i++)
        {
            r45[i] = InverseTh4(points[i], 45, rp);
            r30[i] = InverseTh4(points[i], 30, rp);
            r74[i] = InverseTh4(points[i], 74, rp);
        }

        //Debug.Log($"[{string.Join(", ", r45.Select(v => v.ToString("F6")))}]");
        //Debug.Log($"[{string.Join(", ", r30.Select(v => v.ToString("F6")))}]");
        // 第二步：构造原始列表（包含所有组合）
        var rawList = new List<(Complex x, Complex y)>();

        for (int i = 0; i < count; i++)
        {
            rawList.Add((r30[i], r45[i]));
        }
        // 第三步：过滤无效值（NaN 或 Infinity）
        var validList = rawList
            .Where(t => !(double.IsNaN(t.x.Real) || double.IsInfinity(t.x.Real) ||
                          double.IsNaN(t.x.Imaginary) || double.IsInfinity(t.x.Imaginary) ||
                          double.IsNaN(t.y.Real) || double.IsInfinity(t.y.Real) ||
                          double.IsNaN(t.y.Imaginary) || double.IsInfinity(t.y.Imaginary)))
            .ToList();

        if (validList.Count == 0)
            return validList; // 全部无效，返回空列表

        // 第四步：过滤“极大值”（这里使用模长阈值）
        // 计算所有模长的绝对值，取 99.9% 分位数作为阈值，或直接定义固定阈值（例如 1e10）
        double threshold = 1e10; // 可根据数据调整
        var filteredList = validList
            .Where(t => t.x.Magnitude < threshold && t.y.Magnitude < threshold)
            .ToList();

        if (filteredList.Count == 0)
            return filteredList;

        double maxMagnitude = filteredList
       .SelectMany(t => new[] { t.x.Magnitude, t.y.Magnitude })
       .Max();

        if (maxMagnitude > 0)
        {
            var normalizedList = filteredList
                .Select(t => (x: t.x / maxMagnitude, y: t.y / maxMagnitude))
                .ToList();
            return normalizedList;
        }
        else
        {
            // 若最大模长为 0（全零），则无需归一化，直接返回
            return filteredList;
        }

    }
    public static void glxzf(float jd, float[] s1, float[] s2, float[] s3, float[] s4)
    {

        float d1 = 0;
        float d2 = 0;
        float dd = 0;
       //16 度区间
        if (jd > 164 && jd <180 )
        {
            if (s1[0] > s2[0]) d1 = s1[0] * 100 - (jd - 164); else d1 = s2[0] * 100 - (jd - 164);

            if (s3[0] > s4[0]) d2 = s3[0] * 100 + (jd - 164); else d2 = s4[0] * 100 + (jd - 164);

            dd = (d1 + d2) / 2;
          //s1 < s2 t
           //偏差小 相等
            Debug.Log(dd);
        }
        if(jd < 16 && jd >0)
        {
            // d34 - d12 = 16 -角度差 ?n 
        }
        if (jd > 74 && jd < 90) { }
        //30度区间
        if (jd <30 && jd >16)
        { 
            //d12 /d34 = （30 - 角度差）*2
        }
        //45度区间
        if (jd >45 && jd < 74){  //65左右
         }
        if(jd >30&& jd < 45)
            {   
            //d12 /角度差   d34/角度差
             }
        //74度区间
        if (jd > 90 && jd < 164)
        { //74左右，
        }
    }
    public static (Vector3 F1, Vector3 F2) FitFociByProbability(
    List<Vector3> points,
    float[] trueProb,
    Vector3 initF1,
    Vector3 initF2,
    float a,
    int maxIter = 300,
    float learningRate = 0.001f)
    {
        Vector3 F1 = initF1;
        Vector3 F2 = initF2;

        for (int iter = 0; iter < maxIter; iter++)
        {
            Vector3 gradF1 = Vector3.zero;
            Vector3 gradF2 = Vector3.zero;
            float loss = 0;

            for (int i = 0; i < points.Count; i++)
            {
                Vector3 p = points[i];
                float d1 = Vector3.Distance(p, F1);
                float d2 = Vector3.Distance(p, F2);
                float delta = d1 + d2 - 2f * a;
                float fitProb = Mathf.Exp(-Mathf.Abs(delta) / (2f * a));

                float diff = fitProb - trueProb[i];
                loss += diff * diff;

                float sign = delta >= 0 ? 1f : -1f;
                float coef = diff * fitProb * sign / (2f * a);

                if (d1 > 1e-6f)
                    gradF1 += coef * (p - F1) / d1;
                if (d2 > 1e-6f)
                    gradF2 += coef * (p - F2) / d2;
            }

            F1 -= learningRate * gradF1;
            F2 -= learningRate * gradF2;

            if (loss < 1e-12f) break;
        }

        return (F1, F2);
    }









  


    public static float[] BatchProbability( Vector3 F1, Vector3 F2,List<Vector3>points,float a)
    {
        float[] fff = new float[points.Count];
        for (int i = 0; i < points.Count; i++)
        {
            Vector3 p = points[i];
            float d1 = Vector3.Distance(p, F1);
            float d2 = Vector3.Distance(p, F2);
            float delta = d1 + d2 - 2f * a;
            fff[i] = Mathf.Exp(-Mathf.Abs(delta) /(2*a ));
        }
        return fff;

    }
    public static (Vector3 F1, Vector3 F2) ExtractFoci(
    List<Vector3> points,
    float[] probFocus1, float sigma1,   // 统一 float
    float[] probFocus2, float sigma2)
    {
        int n = points.Count;
        if (n < 4 || probFocus1.Length != n || probFocus2.Length != n)
            throw new System.ArgumentException("点数和概率数组长度不匹配或点数不足4");

        List<float> dist1 = new List<float>(n);
        List<float> dist2 = new List<float>(n);

        for (int i = 0; i < n; i++)
        {
            float p1 = Mathf.Clamp(probFocus1[i], 1e-6f, 1f - 1e-6f);
            float p2 = Mathf.Clamp(probFocus2[i], 1e-6f, 1f - 1e-6f);
            dist1.Add(sigma1 * Mathf.Sqrt(-2f * Mathf.Log(p1)));
            dist2.Add(sigma2 * Mathf.Sqrt(-2f * Mathf.Log(p2)));
        }

        Vector3 F1 = FitFocus(points, dist1);
        Vector3 F2 = FitFocus(points, dist2);
        return (F1, F2);
    }
    public static Vector3 FitFocus(List<Vector3> points, List<float> distances)
    {
        int n = points.Count;
        Vector3 p0 = points[0];
        float d0 = distances[0];

        // 构建线性系统 A * F = b
        var A = Matrix<double>.Build.Dense(n - 1, 3);
        var b = MathNet.Numerics.LinearAlgebra.Vector<double>.Build.Dense(n - 1);

        for (int i = 1; i < n; i++)
        {
            Vector3 diff = points[i] - p0;
            A[i - 1, 0] = 2.0 * diff.x;
            A[i - 1, 1] = 2.0 * diff.y;
            A[i - 1, 2] = 2.0 * diff.z;

            double right = points[i].sqrMagnitude - p0.sqrMagnitude
                           - (distances[i] * distances[i] - d0 * d0);
            b[i - 1] = right;
        }

        // 最小二乘解
        var At = A.Transpose();
        var AtA = At * A;
        var Atb = At * b;
        var solution = AtA.Solve(Atb);

        return new Vector3((float)solution[0], (float)solution[1], (float)solution[2]);
    }
    public static Vector4 pca4(List<(Complex x, Complex y)> yzqx1)
    {
        List<double[]> points4D = yzqx1.Select(item => new double[]
{
    item.x.Real,
    item.x.Imaginary,
    item.y.Real,
    item.y.Imaginary
}).ToList();

        // 2. 构建矩阵：行数 = 样本数，列数 = 4
        int n = points4D.Count;
        int dim = 4;
        var matrix = DenseMatrix.Create(n, dim, (i, j) => points4D[i][j]);

     
        var pca = new PCAScikitLearn();
        pca.Fit(matrix, nComponents: 4);  // 保留所有 4 个主成分
                                          // 第一主成分（4 维向量）
        Vector4 pc1_4D = new Vector4(
            (float)pca.Components[0, 0],
            (float)pca.Components[0, 1],
            (float)pca.Components[0, 2],
            (float)pca.Components[0, 3]
        );
        // 4. 获取结果
        // 主成分（每行是一个主成分，单位向量）
        var components = pca.Components;           // 4×4 矩阵（若 nComponents=4）
                                                   // 解释方差比例
        var varianceRatio = pca.ExplainedVarianceRatio;  // 长度为 4 的向量
                                                         // 第一主成分（方向）
        var pc1 = new double[4];




        for (int j = 0; j < 4; j++) pc1[j] = components[0, j];

        // 输出
        Debug.Log(pc1_4D);
        return pc1_4D;
    }
    public static Vector3 pca(List<Vector3> points)
    {
        double[] r = new double[3];
        int dim = 3;
        var matrix = DenseMatrix.Create(points.Count, dim, (i, j) =>
        {
            Vector3 p = points[i];
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
        float fj = Mathf.Atan2(pc1.z,pc1.x) * Mathf.Rad2Deg;
        float d2 = Mathf.Acos(pc1.y) * Mathf.Rad2Deg;
        Vector3 v3 = new Vector3(
              Mathf.Sin(d2 * Mathf.Deg2Rad) * Mathf.Cos(fj * Mathf.Deg2Rad),
              Mathf.Cos(d2 * Mathf.Deg2Rad),
              Mathf.Sin(d2 * Mathf.Deg2Rad) * Mathf.Sin(fj * Mathf.Deg2Rad)
          );
        return v3;

    }
    


    public static float ComputeRp(List<Vector3> points)
    {
        if (points.Count == 0) return 0f;

        float sum = 0f;
        foreach (var p in points)
        {
            sum += p.magnitude; // sqrt(x² + y² + z²)
        }
        float rp = sum / points.Count;

        
        return rp;
    }



    public static (Complex tau1, Complex tau2) ComputeTaus()
    {
        Complex K2 = Carlsonfk.K(2.0); // K(2)
       
        Complex Kprime2 = Carlsonfk.K(Complex.Sqrt(1 - 2.0 * 2.0)); // K(i√3)
        
        Complex tau1 = Complex.ImaginaryOne * Kprime2 / K2;


        Complex Ksqrt2 = Carlsonfk.K(Math.Sqrt(2)); // K(√2)
       
        Complex KprimeSqrt2 = Carlsonfk.K(Complex.Sqrt(1 - 2.0)); // K(i)
      
        Complex tau2 = Complex.ImaginaryOne * KprimeSqrt2 / Ksqrt2;

        if (tau1.Imaginary < 0) tau1 = -tau1;
        if (tau2.Imaginary < 0) tau2 = -tau2;
      
        return (tau1, tau2);
    }

    /// <summary>
    /// 软最小距离: softmin(z) = -γ·ln(Σ_λ exp(-|z-λ|/γ))，数值稳定形式
    ///   = minDist - γ·ln(Σ exp(-(dist-minDist)/γ))。
    /// γ 自适应: γ = gammaFraction × scale，
    ///   scale 默认 = 最近两格点距离间隙 dists[1]-dists[0]（≈格点间距，
    ///   即"点云到格点距离"的自然尺度；minDist≈0 时退化为格点间距，仍有效）；
    ///   也可外部传入(如点云级 σ)覆盖。
    ///   夹取到 [γFloor·scale, scale]：γ 过大 → softmin 退化为均值(梯度消失)；
    ///   γ 过小 → exp 下溢/退化为硬 min(数值不稳定)。
    /// </summary>
    private static double SoftMinDistance(Complex z, List<Complex> lattice,
                                          double gammaFraction = 0.1,
                                          double scale = -1.0)
    {
        int n = lattice.Count;
        double[] dists = new double[n];
        double minDist = double.MaxValue;
        for (int i = 0; i < n; i++)
        {
            double dist = (z - lattice[i]).Magnitude;
            dists[i] = dist;
            if (dist < minDist) minDist = dist;
        }

        // 尺度: 外部提供(点云级 σ) 或内部自适应(最近格点间隙 ≈ 格点间距)
        if (scale <= 0)
        {
            Array.Sort(dists);
            scale = (n > 1) ? Math.Max(dists[1] - dists[0], 1e-12) : 1e-12;
        }

        double gamma = Math.Max(1e-6 * scale,
                                Math.Min(gammaFraction * scale, scale));
        double sum = 0.0;
        for (int i = 0; i < n; i++)
            sum += Math.Exp(-(dists[i] - minDist) / gamma);
        return minDist - gamma * Math.Log(sum);
    }





    public static Complex InverseTh4(Vector3 vs3, float d2,float rp)
    {
        d2 = d2 * Mathf.Deg2Rad;
        // 1. 计算 rs, zs, phis
        float rs = Mathf.Sqrt(vs3.x * vs3.x + vs3.y * vs3.y);
        float zs = vs3.z;
        float phis = Mathf.Atan2(vs3.y, vs3.x);
        if (phis < 0) phis += 2 * Mathf.PI;

        // 2. 计算 r 和 thetas
        float r = Mathf.Sqrt(rs * rs + zs * zs);
        float sin_d2 = Mathf.Sin(d2);
        float cos_d2 = Mathf.Cos(d2);
        // 验证：rs = r * sin_d2, zs = r * cos_d2，若不一致则可能有误差
        // 但假设数据一致，直接由 phis 求 thetas
        float thetas = phis / sin_d2;
        // 将 thetas 归一化到 [0, 2π)
        while (thetas < 0) thetas += 2 * Mathf.PI;
        while (thetas >= 2 * Mathf.PI) thetas -= 2 * Mathf.PI;

        // 3. 计算 Rs
        float Rs = Mathf.Pow(r/rp, 1.0f / sin_d2);

        // 4. 构造 uvs
        float u = Rs * Mathf.Cos(thetas);
        float v = Rs * Mathf.Sin(thetas);

        return new Complex(u, v);
    }

    
    public static double Gamma(double x)
    {
        double[] p = { 676.5203681218851, -1259.1392167224028, 771.32342877765313,
                   -176.61502916214059, 12.507343278686905, -0.13857109526572012,
                   9.9843695780195716e-6, 1.5056327351493116e-7 };
        if (x < 0.5)
        {
            return Math.PI / (Math.Sin(Math.PI * x) * Gamma(1 - x));
        }
        x -= 1.0;
        double a = 0.99999999999980993;
        for (int i = 0; i < p.Length; i++) a += p[i] / (x + i + 1.0);
        double t = x + p.Length - 0.5;
        return Math.Sqrt(2 * Math.PI) * Math.Pow(t, x + 0.5) * Math.Exp(-t) * a;
    }
}
public static class LatticeAnalysis
{
    /// <summary>
    /// 从复数对列表中估计格基。返回 4x4 实矩阵 B，其列向量为格基。
    /// 假设数据点完全位于一个格上（无噪声），选取前 4 个线性独立的向量作为基。
    /// </summary>
    /// 
    private static IEnumerable<int[]> GetCombinations(int[] elements, int k)
    {
        int n = elements.Length;
        if (k < 0 || k > n) yield break;

        int[] current = new int[k];
        for (int i = 0; i < k; i++) current[i] = i;

        while (true)
        {
            yield return current.Select(idx => elements[idx]).ToArray();

            // 寻找下一个组合
            int pos = k - 1;
            while (pos >= 0 && current[pos] == n - k + pos)
                pos--;

            if (pos < 0) break;

            current[pos]++;
            for (int i = pos + 1; i < k; i++)
                current[i] = current[i - 1] + 1;
        }
    }
    public static Matrix<double> EstimateLatticeBasis(List<(Complex x, Complex y)> points)
    {
        int n = points.Count;
        if (n < 4) throw new ArgumentException("至少需要 4 个点来估计格基");

        // 将点转换为 4 维实向量
        var vectors = points.Select(p => MathNet.Numerics.LinearAlgebra.Vector<double>.Build.Dense(new double[]
        {
            p.x.Real, p.x.Imaginary, p.y.Real, p.y.Imaginary
        })).ToList();

        // 寻找一组线性独立的向量（最多 4 个）
        var basisList = new List<MathNet.Numerics.LinearAlgebra.Vector<double>>();
        foreach (var v in vectors)
        {
            if (basisList.Count == 4) break;

            if (basisList.Count == 0)
            {
                basisList.Add(v);
                continue;
            }

            // 构造矩阵测试秩
            var mat = DenseMatrix.OfColumnVectors(basisList.Concat(new[] { v }));
            if (mat.Rank()> basisList.Count)
                basisList.Add(v);
        }

        if (basisList.Count < 4)
            throw new Exception("无法找到 4 个线性独立的向量，可能数据点不够或存在退化");

        // 返回以这些向量为列的矩阵
        return DenseMatrix.OfColumnVectors(basisList);
    }

    
    public static Complex[,] ExtractPeriodMatrix(Matrix<double> B)
    {
        if (B.RowCount != 4 || B.ColumnCount != 4)
            throw new ArgumentException("B 必须是 4x4 矩阵");

        // 构建 2x4 复数矩阵 C，列 j 为 [B[0,j] + i*B[1,j], B[2,j] + i*B[3,j]]
        var C = Matrix<Complex>.Build.Dense(2, 4);
        for (int j = 0; j < 4; j++)
        {
            Complex z1 = new Complex(B[0, j], B[1, j]);
            Complex z2 = new Complex(B[2, j], B[3, j]);
            C[0, j] = z1;
            C[1, j] = z2;
        }

        // 尝试所有 2 列组合，寻找可逆的 A
        int[] columnIndices = { 0, 1, 2, 3 };
        var combinations = GetCombinations(columnIndices, 2);
        foreach (var cols in combinations)
        {
            // 剩余的两列索引
            var remaining = columnIndices.Except(cols).ToArray();

            // 构建 A
            var A = Matrix<Complex>.Build.Dense(2, 2);
            A[0, 0] = C[0, cols[0]];
            A[0, 1] = C[0, cols[1]];
            A[1, 0] = C[1, cols[0]];
            A[1, 1] = C[1, cols[1]];

            Complex det = A.Determinant();
            if (Complex.Abs(det) < 1e-12)
                continue; // A 奇异，尝试下一组

            // 计算 A 的逆
            var Ainv = A.Inverse();

            // 计算 Ω = Ainv * C_remaining (2x2)
            var C_rem = Matrix<Complex>.Build.Dense(2, 2);
            for (int r = 0; r < 2; r++)
            {
                for (int c = 0; c < 2; c++)
                {
                    C_rem[r, c] = C[r, remaining[c]];
                }
            }

            var Omega = Ainv * C_rem;

            // 转换为 Complex[,] 返回
            Complex[,] result = new Complex[2, 2];
            for (int r = 0; r < 2; r++)
                for (int c = 0; c < 2; c++)
                    result[r, c] = Omega[r, c];

            return result;
        }

        throw new Exception("无法从格基中提取周期矩阵：所有 2x2 子矩阵均奇异");
    }
}


public static class LLL
{
  
    public static Matrix<double> Reduce(Matrix<double> basis, double delta = 0.75)
    {
        int n = basis.RowCount; // 向量维度
        int m = basis.ColumnCount; // 基向量个数

        if (m > n) throw new ArgumentException("基向量个数不能超过维度");

        // 将列向量转为数组方便操作
        MathNet.Numerics.LinearAlgebra.Vector<double>[] B = new MathNet.Numerics.LinearAlgebra.Vector<double>[m];
        for (int i = 0; i < m; i++)
            B[i] = basis.Column(i);

        // Gram-Schmidt 正交化系数
        double[,] mu = new double[m, m];
        double[] BnormSq = new double[m]; // ||B*_i||^2

        // 初始 Gram-Schmidt
        ComputeGS(B, mu, BnormSq);

        int k = 1; // 当前处理的下标（从 0 开始，但算法中 k 指第二个向量）
        while (k < m)
        {
            // 尺寸约简：对 j = k-1 down to 0
            for (int j = k - 1; j >= 0; j--)
            {
                if (Math.Abs(mu[k, j]) > 0.5)
                {
                    double q = Math.Round(mu[k, j]);
                    B[k] = B[k] - q * B[j];
                    // 更新 mu 和 BnormSq
                    UpdateGS(B, mu, BnormSq, k, j, q);
                }
            }

            // Lovász 条件
            if (BnormSq[k] >= (delta - mu[k, k - 1] * mu[k, k - 1]) * BnormSq[k - 1])
            {
                k++;
            }
            else
            {
                // 交换 B[k] 和 B[k-1]
                MathNet.Numerics.LinearAlgebra.Vector<double> temp = B[k];
                B[k] = B[k - 1];
                B[k - 1] = temp;

                // 重新计算受影响的 Gram-Schmidt 系数
                RecomputeGS(B, mu, BnormSq, Math.Max(0, k - 2));
                k = Math.Max(1, k - 1);
            }
        }

        // 构造结果矩阵
        Matrix<double> result = DenseMatrix.Create(n, m, (i, j) => B[j][i]);
        return result;
    }
    private static List<MathNet.Numerics.LinearAlgebra.Vector<double>> bstar =  new List<MathNet.Numerics.LinearAlgebra.Vector<double>>();
    // 计算 Gram-Schmidt 正交化（从 start 开始到末尾）
    private static void ComputeGS(MathNet.Numerics.LinearAlgebra.Vector<double>[] B, double[,] mu, double[] BnormSq, int start = 0)
    {
       
        for (int i = 0; i < B.Length; i++)
        {
            bstar.Add(null);
        }
        for (int i = start; i < B.Length; i++)
        {
            MathNet.Numerics.LinearAlgebra.Vector<double> b_i = B[i];
            MathNet.Numerics.LinearAlgebra.Vector<double> b_i_orth = b_i.Clone();

            for (int j = 0; j < i; j++)
            {
                if (BnormSq[j] < 1e-20) continue;
                mu[i, j] = b_i.DotProduct(bstar[j]) / BnormSq[j]; // 使用 bstar[j]
                b_i_orth -= mu[i, j] * bstar[j];
            }
            bstar[i] = b_i_orth; // 保存当前正交化向量
            BnormSq[i] = b_i_orth.DotProduct(b_i_orth);
            if (BnormSq[i] < 1e-20) BnormSq[i] = 0;
        }
        
    }


    // 更新 GS 系数（当 B[k] 减去 q*B[j] 时）
    private static void UpdateGS(MathNet.Numerics.LinearAlgebra.Vector<double>[] B, double[,] mu, double[] BnormSq, int k, int j, double q)
    {
        // B[k] 已经更新为 B[k] - q*B[j]
        // 重新计算 mu[k, j] 及其后所有列
        for (int i = 0; i <= j; i++)
        {
            if (BnormSq[i] < 1e-20) continue;
            mu[k, i] = B[k].DotProduct(B[i]) / BnormSq[i];
        }
        // 对于 i > j 的 mu[k, i] 需要更新吗？实际上只影响 k 行，其他行不变
        // 但为了简单，重新计算整个 k 行的 mu 和 BnormSq[k]
        MathNet.Numerics.LinearAlgebra.Vector<double> b_k_orth = B[k].Clone();
        for (int i = 0; i < k; i++)
        {
            if (BnormSq[i] < 1e-20) continue;
            mu[k, i] = B[k].DotProduct(B[i]) / BnormSq[i];
            b_k_orth -= mu[k, i] * B[i];
        }
        BnormSq[k] = b_k_orth.DotProduct(b_k_orth);
        if (BnormSq[k] < 1e-20) BnormSq[k] = 0;
    }

    // 交换后重新计算 GS（从 start 开始）
    private static void RecomputeGS(MathNet.Numerics.LinearAlgebra.Vector<double>[] B, double[,] mu, double[] BnormSq, int start)
    {
        // 重新计算 start 及之后所有向量的 GS 系数
        ComputeGS(B, mu, BnormSq, start);
    }
}


public static class ComplexLinearFit
{
    /// <summary>
    /// 拟合 z ≈ a * w + b，返回 a, b 和最大残差。
    /// </summary>
    public static void Fit(Complex[] z, Complex[] w, out Complex a, out Complex b, out double maxError)
    {
        if (z.Length != w.Length || z.Length == 0)
            throw new ArgumentException("输入数组长度必须相同且非空");

        int N = z.Length;

        // 构造实数最小二乘：未知数 [aR, aI, bR, bI]
        // 方程: (aR + i aI)(wR + i wI) + (bR + i bI) = zR + i zI
        // 实部: aR*wR - aI*wI + bR = zR
        // 虚部: aR*wI + aI*wR + bI = zI
        double[,] A = new double[2 * N, 4];
        double[] y = new double[2 * N];

        for (int i = 0; i < N; i++)
        {
            double wR = w[i].Real, wI = w[i].Imaginary;
            double zR = z[i].Real, zI = z[i].Imaginary;

            // 实部方程
            A[2 * i, 0] = wR;
            A[2 * i, 1] = -wI;
            A[2 * i, 2] = 1;
            A[2 * i, 3] = 0;
            y[2 * i] = zR;

            // 虚部方程
            A[2 * i + 1, 0] = wI;
            A[2 * i + 1, 1] = wR;
            A[2 * i + 1, 2] = 0;
            A[2 * i + 1, 3] = 1;
            y[2 * i + 1] = zI;
        }

        // 正规方程 A^T A x = A^T y
        double[,] AtA = new double[4, 4];
        double[] Aty = new double[4];

        for (int r = 0; r < 4; r++)
        {
            for (int c = 0; c < 4; c++)
            {
                double sum = 0;
                for (int k = 0; k < 2 * N; k++)
                    sum += A[k, r] * A[k, c];
                AtA[r, c] = sum;
            }
            double sum2 = 0;
            for (int k = 0; k < 2 * N; k++)
                sum2 += A[k, r] * y[k];
            Aty[r] = sum2;
        }

        // 解 4x4 线性方程组
        double[] solution = SolveLinearSystem(AtA, Aty);

        a = new Complex(solution[0], solution[1]);
        b = new Complex(solution[2], solution[3]);

        // 计算最大残差（模长）
        maxError = 0;
        for (int i = 0; i < N; i++)
        {
            Complex predicted = a * w[i] + b;
            double error = (predicted - z[i]).Magnitude;
            if (error > maxError) maxError = error;
        }
    }

    // 高斯消元解线性方程组（4x4）
    private static double[] SolveLinearSystem(double[,] A, double[] b)
    {
        int n = 4;
        double[,] aug = new double[n, n + 1];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
                aug[i, j] = A[i, j];
            aug[i, n] = b[i];
        }

        for (int i = 0; i < n; i++)
        {
            // 选主元
            int maxRow = i;
            for (int k = i + 1; k < n; k++)
                if (Math.Abs(aug[k, i]) > Math.Abs(aug[maxRow, i]))
                    maxRow = k;
            for (int j = i; j <= n; j++)
            {
                double tmp = aug[i, j];
                aug[i, j] = aug[maxRow, j];
                aug[maxRow, j] = tmp;
            }

            if (Math.Abs(aug[i, i]) < 1e-12) continue;

            for (int k = i + 1; k < n; k++)
            {
                double factor = aug[k, i] / aug[i, i];
                for (int j = i; j <= n; j++)
                    aug[k, j] -= factor * aug[i, j];
            }
        }

        double[] x = new double[n];
        for (int i = n - 1; i >= 0; i--)
        {
            x[i] = aug[i, n];
            for (int j = i + 1; j < n; j++)
                x[i] -= aug[i, j] * x[j];
            x[i] /= aug[i, i];
        }
        return x;
    }
}

public static class LatticeRansac
{
    /// <summary>
    /// 使用 RANSAC 寻找最大的格内点子集。
    /// </summary>
    /// <param name="points">复数对列表，每个点将被转为4D实向量。</param>
    /// <param name="threshold">残差阈值，小于该值认为是内点。</param>
    /// <param name="iterations">RANSAC 迭代次数。</param>
    /// <returns>返回内点索引集合和最佳格基（列向量为基）。</returns>
    public static (List<int> inlierIndices, Matrix<double> bestBasis) FindLargestLattice(
        List<(Complex x, Complex y)> points,
        double threshold = 1e-5,
        int iterations = 1000)
    {
        int n = points.Count;
        if (n < 4) throw new ArgumentException("至少需要 4 个点");

        // 转换为 4D 向量
        var vectors = points.Select(p => MathNet.Numerics.LinearAlgebra.Vector<double>.Build.Dense(new double[]
        {
            p.x.Real, p.x.Imaginary, p.y.Real, p.y.Imaginary
        })).ToList();

        var random = new System.Random();
        List<int> bestInliers = new List<int>();
        Matrix<double> bestBasis = null;

        for (int iter = 0; iter < iterations; iter++)
        {
            // 随机选 4 个不同索引
            var indices = Enumerable.Range(0, n).OrderBy(x => random.Next()).Take(4).ToArray();
            // 检查线性无关性
            var candidateVectors = indices.Select(idx => vectors[idx]).ToArray();
            var B = DenseMatrix.OfColumnVectors(candidateVectors);
            if (B.Rank() < 4) continue; // 退化，跳过

            // 求逆矩阵
            Matrix<double> invB;
            try { invB = B.Inverse(); }
            catch { continue; }

            // 统计内点
            var inliers = new List<int>();
            for (int i = 0; i < n; i++)
            {
                var c_real = invB * vectors[i];
                var c_int = MathNet.Numerics.LinearAlgebra.Vector<double>.Build.Dense(4, j => Math.Round(c_real[j]));
                var reconstructed = B * c_int;
                double error = (reconstructed - vectors[i]).L2Norm();
                if (error < threshold)
                {
                    inliers.Add(i);
                }
            }

            if (inliers.Count > bestInliers.Count)
            {
                bestInliers = inliers;
                bestBasis = B;
            }
        }

        // 可选：用最佳内点重新估计基（例如 LLL 归约）
        if (bestInliers.Count >= 4)
        {
            var inlierVectors = bestInliers.Select(i => vectors[i]).ToList();
            // 重新用前 4 个线性无关的内点作为基，然后 LLL
            var basisList = new List<MathNet.Numerics.LinearAlgebra.Vector<double>>();
            foreach (var v in inlierVectors)
            {
                if (basisList.Count == 4) break;
                var temp = basisList.Concat(new[] { v });
                var mat = DenseMatrix.OfColumnVectors(temp);
                if (mat.Rank() > basisList.Count)
                    basisList.Add(v);
            }
            if (basisList.Count == 4)
            {
                var B0 = DenseMatrix.OfColumnVectors(basisList);
                // 假设你已实现 LLL.Reduce
                // bestBasis = LLL.Reduce(B0, 0.99);
                bestBasis = B0; // 暂时用原始基
            }
        }

        return (bestInliers, bestBasis);
    }
}






    /// </summary>
 

public static class Carlsonfk
{
    public static Complex K(Complex k)
    {
            return RF(Complex.Zero, 1 - k*k, Complex.One);   
    }
    public static Complex RF(Complex x, Complex y, Complex z)
    {
        Complex X = 0;
        Complex Y = 0;
        Complex Z = 0;
        Complex u = 0;
        double eps = 1e-12;
        for (int i = 0;i < 300; i++)
        {
            u = (x + y + z) / 3.0;
            X = (u - x) / u;
            Y = (u - y) / u;
            Z = (u - z) / u;

          double MAX = Math.Max(Math.Max(Complex.Abs(X), Complex.Abs(Y)),
      Complex.Abs(Z)); if (MAX < eps * Complex.Abs(u)) break;
            Complex sx = Complex.Sqrt(x);
            Complex sy = Complex.Sqrt(y);
            Complex sz = Complex.Sqrt(z);
            Complex lambda = sx * sy + sx * sz + sy * sz;

            x = (x + lambda) / 4.0;
            y = (y + lambda) / 4.0;
            z = (z + lambda) / 4.0;
        }
        Complex E2 = X * Y - Z * Z;
        Complex E3 = X * Y * Z;
        Complex v = (1 + E2 * (E2 / 24 - E3 * 3.0 / 44 - 0.1) + E3 / 14) / Complex.Sqrt(u);
        return v;
    }
    public static Complex RD(Complex x, Complex y, Complex z)
    {
        double eps = Math.Sqrt(2.220446049250313e-16);
        if (x ==y)
        {
            Complex sx = Complex.Sqrt(x);
            Complex sy = Complex.Sqrt(y);
            Complex sum = 0;
            Complex pow = 0.25f;
            for(int i =0;i<300;i++)
            {
                if (Complex.Abs(sx-sy) <=2.7 * eps * Complex.Abs(sx)) break;
                Complex t = Complex.Sqrt(sx *sy);
                sx = (sx + sy) / 2;
                sy = t;
                pow *= 2;
                sum += pow * (sx - sy) * (sx - sy);

            }
            Complex rf = Math.PI / (sx + sy);
            Complex pt = (sx + 3.0 * sy) / (4 * z * (sx + sy));
            pt -= sum / (z * (y - z));

            return pt*rf*3;
        }
        Complex An = (x + y + 3 * z) / 5;
        Complex A0 = An;
        Complex q = Complex.Pow(Math.E / 4.0, -1.0 / 8.0);
        Complex k = 0;
        Complex fn = 1;
        Complex RD_sum = 0;
        for(int i = 0; i < 300; i++)
        {
           
            Complex sx = Complex.Sqrt(x);
            Complex sy = Complex.Sqrt(y);
            Complex sz = Complex.Sqrt(z);
            Complex lambda = sx * sy + sx * sz + sy * sz;
            RD_sum += fn / (sz * (z + lambda));
            An = (An + lambda) / 4;
            x = (x + lambda) / 4;
            y = (y + lambda) / 4;
            z = (z + lambda) / 4;
            fn /= 4;
            q /= 4;
           
            if ( q.Real < An.Real) break;
        }
        Complex X = fn * (A0 - x) / An;
        Complex Y = fn * (A0 - y) / An;
        Complex Z = -(X + Y) / 3;
        Complex E2 = X * Y - 6 * Z * Z;
        Complex E3 = (3 * X * Y - 8 * Z * Z) * Z;
        Complex E4 = 3 * (X * Y - Z * Z) * Z * Z;
        Complex E5 = X * Y * Z * Z * Z;
        Complex result = fn * Complex.Pow(An, -3.0 / 2) *
               (1 - 3 * E2 / 14 + E3 / 6 + 9 * E2 * E2 / 88 -
                3 * E4 / 22 - 9 * E2 * E3 / 52 + 3 * E5 / 26 -
                E2 * E2 * E2 / 16 + 3 * E3 * E3 / 40 +
                3 * E2 * E4 / 20 + 45 * E2 * E2 * E3 / 272 -
                9 * (E3 * E4 + E2 * E5) / 68);
        result += 3 * RD_sum;
        return result;
    }

    public static Complex E(Complex k)
    {
        // E(k) = ∫_0^{π/2} sqrt(1 - k^2 sin^2 θ) dθ
        return Integrate(theta =>
        {
            Complex sin = Complex.Sin(theta);
            return Complex.Sqrt(1 - k * k * sin * sin);
        }, 0.0, Math.PI / 2.0, 2000);
    }
    public static Complex RD1z(Complex k)
    {
       
        Complex Kk = K(k);   // 你已有的 K(k)
        Complex Ek = E(k);   // 你已有的 E(k)（数值积分或 Carlson）

        return (3.0 / (k * k)) * (Kk - Ek);
    }
    public static Complex RDz1(Complex k)
    {

        Complex Kk = K(k);   // 你已有的 K(k)
        Complex Ek = E(k);   // 你已有的 E(k)（数值积分或 Carlson）
        return (Ek * 3.0 / (1.0 - k * k) - RD1z(k));
    }
    //return K(k) - 1.0 / 3.0 * k* k *RD(Complex.Zero, 1 - k * k, Complex.One);
    //return 1.0/3.0(1.0-k*K)[RD(0,1-k*k,1)+RD[0,1,1-k*K]] k=0?
    // K(k)=RD(0,1-k*k,1)+1/3(1-k*k)RD(0,1,1-k*k)
    

   
    private static Complex Integrate(Func<double, Complex> f, double a, double b, int n)
    {
        if (n % 2 != 0) n++;
        double h = (b - a) / n;
        Complex sum = f(a) + f(b);
        for (int i = 1; i < n; i++)
        {
            double x = a + i * h;
            sum += (i % 2 == 0) ? 2 * f(x) : 4 * f(x);
        }
        return sum * h / 3.0;
    }

    public static Complex DK(Complex k)
    {
       
        return (E(k) - (1-k*k) * K(k)) / (k * (1-k*k));
    }
  
    public static Complex NK(Complex aot)
    {
        Complex k = 0;
        double h = 1e-6;
        if (aot.Real < 0)
        {
            aot = Complex.Sqrt(Complex.Pow(aot, 2));
            Debug.Log(" 实数为负");
        }
        for (int i = 0; i < 500; i++)
        {
            Complex f = K(k) - aot;
            if (f.Magnitude < 1e-12) break;
            Complex df = (K(k + h) - K(k - h)) / (2.0 * h);
            if (df.Magnitude < 1e-14) { k += new Complex(0.01, 0.01); continue; }
            Complex step = f / df;
            double stepMag = step.Magnitude;
            double maxStep = 0.2;
            if (stepMag > maxStep) step *= maxStep / stepMag;
            k -= step;
        }
            return k;
        
    }


   
}



