using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Complex = System.Numerics.Complex;
using Vector3 = UnityEngine.Vector3;
using Vector4 = UnityEngine.Vector4;
using MathNet.Numerics.LinearAlgebra;
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
      
        float e3 = (float)(Math.Cos(d2) * 2 / (1 + Math.Sin(d2)));
        float h = rp * Mathf.Cos(d2);
        float c = h * e3;



        List<(Complex x, Complex y)> yzqx1 = yzqx(points, out r45,out r30, out r74, rp);
        
        
       
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
        double tt = Dist(t1, t2)*Math.Sqrt(2);
       
        double tt1 = Dist1(t1,t2,t11,t22);
        


       


        Complex K2 = Carlsonfk.K(2.0);
        Complex Kprime2 = Carlsonfk.K(Complex.Sqrt(1 - 4.0));
        Complex tau1 = Complex.ImaginaryOne * Kprime2 / K2;      // 模参数 tau = i*K'/K

        Complex Ksqrt2 = Carlsonfk.K(Math.Sqrt(2));
        Complex KprimeSqrt2 = Carlsonfk.K(Complex.Sqrt(1 - 2.0));
        Complex tau2 = Complex.ImaginaryOne * KprimeSqrt2 / Ksqrt2;   // 模参数 tau = i*K'/K

    ;
     









    }
    public static double Dist(Complex a, Complex b)
    {
        double ya = a.Imaginary;
        double yb = b.Imaginary;

        if (ya <= 0 || yb <= 0)
            throw new ArgumentException(
                $"双曲距离要求 Im>0，实际 Im(a)={ya}, Im(b)={yb}");

        double diff2 = (a - b).Magnitude;
        diff2 *= diff2;

        double arg = 1.0 + diff2 / (2.0 * ya * yb);

        // 数值保护：arccosh 定义域 [1, ∞)
        if (arg < 1.0) arg = 1.0;

        return Math.Acosh(arg);
    }
    public static double Dist1(Complex t1, Complex t2,
        Complex t1p, Complex t2p)
    {
        double d1 = Dist(t1, t1p);
        double d2 = Dist(t2, t2p);
        return Math.Sqrt(d1 * d1 + d2 * d2);
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
       
        var rawList = new List<(Complex x, Complex y)>(count);
        for (int i = 0; i < count; i++)
        {
            rawList.Add((r30[i], r45[i]));
        }

        // 第三步：过滤无效值（NaN / Infinity）
        var validList = rawList
            .Where(t =>
                !(double.IsNaN(t.x.Real) || double.IsInfinity(t.x.Real) ||
                  double.IsNaN(t.x.Imaginary) || double.IsInfinity(t.x.Imaginary) ||
                  double.IsNaN(t.y.Real) || double.IsInfinity(t.y.Real) ||
                  double.IsNaN(t.y.Imaginary) || double.IsInfinity(t.y.Imaginary)))
            .ToList();

        if (validList.Count == 0)
            return validList;

        // 第四步：过滤极大值（用模长阈值，不做缩放）
        const double threshold = 1e10; // 按需调整
        var filteredList = validList
            .Where(t => t.x.Magnitude < threshold && t.y.Magnitude < threshold)
            .ToList();

        // ★ 不要做归一化：直接返回
        return filteredList;
    


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
    public static Vector3 FitFocus(System.Collections.Generic.List<Vector3> points,
                               System.Collections.Generic.List<float> distances)
    {
        int n = points.Count;
        Vector3 p0 = points[0];
        float d0 = distances[0];

        // 构建线性系统 A * F = b
        var A = Matrix<double>.Build.Dense(n - 1, 3);
        var b = Vector<double>.Build.Dense(n - 1);

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

        // 最小二乘解：正规方程 (AᵀA) F = Aᵀb
        var At = A.Transpose();
        var AtA = At * A;      // 3×3
        var Atb = At * b;      // 3×1

        // 求解线性方程组
        var solution = AtA.Solve(Atb);

        return new Vector3(
            (float)solution[0],
            (float)solution[1],
            (float)solution[2]
        );
    }

    public static  Vector3 pca( List<Vector3> points)
    {
        int dim = 3;

        int n = points.Count;
        var matrix = Matrix<double>.Build.Dense(n, dim);

        for (int i = 0; i < n; i++)
        {
            Vector3 p = points[i];
            for (int j = 0; j < dim; j++)
            {
                if (j == 0) matrix[i, j] = p.x;
                else if (j == 1) matrix[i, j] = p.y;
                else if (j == 2) matrix[i, j] = p.z;
                else matrix[i, j] = 0.0;   // 超出 3 维补 0
            }
        }
        var pca = new PCAScikitLearn();
        pca.Fit(matrix, nComponents: 3);
        Vector3 pc1 = new Vector3(
    (float)pca.Components[0, 0],
    (float)pca.Components[0, 1],
    (float)pca.Components[0, 2]);
        float absX = Mathf.Abs(pc1.x);
        float absY = Mathf.Abs(pc1.y);
        float absZ = Mathf.Abs(pc1.z);
        float max = Mathf.Max(absX, absY, absZ);
       

        return pc1;
    }

    public static Vector4 pca4(List<(Complex x, Complex y)> yzqx1)
    {
        // 1. 转为 4D 点集
        int n = yzqx1.Count;
        int dim = 4;

        // 2. 构建矩阵：行数 = 样本数，列数 = 4
        var matrix = Matrix<double>.Build.Dense(n, dim);
        for (int i = 0; i < n; i++)
        {
            var item = yzqx1[i];
            matrix[i, 0] = item.x.Real;
            matrix[i, 1] = item.x.Imaginary;
            matrix[i, 2] = item.y.Real;
            matrix[i, 3] = item.y.Imaginary;
        }

        // 3. PCA 拟合
        var pca = new PCAScikitLearn();
        pca.Fit(matrix, nComponents: 4);

        // 4. 取第一主成分
        var components = pca.Components;   // 4×4 矩阵，每行一个主成分

        Vector4 pc1_4D = new Vector4(
            (float)components[0, 0],
            (float)components[0, 1],
            (float)components[0, 2],
            (float)components[0, 3]
        );

        Debug.Log(pc1_4D);
        return pc1_4D;
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
        
        Complex tau1 = Complex.ImaginaryOne * Kprime2 / K2;      // 模参数 tau = i*K'/K


        Complex Ksqrt2 = Carlsonfk.K(Math.Sqrt(2)); // K(√2)
       
        Complex KprimeSqrt2 = Carlsonfk.K(Complex.Sqrt(1 - 2.0)); // K(i)
      
        Complex tau2 = Complex.ImaginaryOne * KprimeSqrt2 / Ksqrt2;   // 模参数 tau = i*K'/K

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






    /// </summary>
 

