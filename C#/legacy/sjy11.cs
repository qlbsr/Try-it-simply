using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using static UnityEngine.GraphicsBuffer;
using Vector2 = UnityEngine.Vector2;
using Vector3 = UnityEngine.Vector3;



// ---------- test 类完整流程 ----------
class test : MonoBehaviour
{
    public sjy sjy11;
    void Start()
    {
        // 1. 读取点云
        VectorList vectorList = JsonVectorParser.jsonpy("pyjson");
        List<Vector3> pointCloud = vectorList.Vector3List;
        if (pointCloud == null || pointCloud.Count == 0)
        {
            Debug.LogError("点云为空");
            return;
        }

        // 2. 初始化几何处理器
        sjy11 = new sjy();
        sjy11.vector3s = pointCloud;

        // 3. 计算全局半顶角 d2 和主轴方向 v3（通过 PCA）
        float d2_global = sjy11.jh(1);
        float fj = sjy11.pca();   // 触发 PCA，结果存储于 sjy.r 和 sjy.v3

        // 4. 生成扇区点集，并计算母线长 rp 和主轴 v3
        Vector3[][] vecs1 = sjy11.dn();
        Vector3[][] vectors2 = sjy11.sx(vecs1);
        // 此时 sjy.rp, sjy.v3 已被设置

        // 5. 取第一组点集（圆上交点）作为基准点
        List<Vector3> points = new List<Vector3>(vectors2[0]);
        if (points == null || points.Count == 0)
        {
            Debug.LogError("基准点集为空");
            return;
        }

        // 6. 计算椭球吻合概率
        float[] probs = sjy11.ComputeProbabilities(points, d2_global, sjy11.v3);
        if (probs == null || probs.Length == 0)
        {
            Debug.LogError("概率数组为空");
            return;
        }

        // 7. 生成叶平面映射，填充 uvss 列表
        Vector2[] uvs = sjy11.MapDoubleConeToLeaf(points);
        // 检查 uvss 是否已填充
        if (sjy11.uvss == null || sjy11.uvss.Count == 0)
        {
            Debug.LogError("uvss 未填充，请检查 MapDoubleConeToLeaf 实现");
            return;
        }

        sjy11.xl1(sjy11.v3, points, probs);
        // 8. 计算色散向量 vss
        sjy11.Vss1(sjy11.vss, out Vector3 V, out Vector3 V_A, out Vector3 V_D,
                 out Vector3 V_q1, out Vector3 V_q2);
        sjy11.rbf(sjy11.points1, sjy11.points2,probs);


    }

    










}
