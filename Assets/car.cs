using UnityEngine;
using UnityEngine.UI;

[ExecuteInEditMode]
[RequireComponent(typeof(Camera))]
public class EdgeDetectionEffect : MonoBehaviour
{
    [Header("检测参数")]
    [Range(0.0f, 1.0f)]
    public float edgeThreshold = 0.2f;      // 边缘灵敏度，越小边缘越多

    [Header("轮廓样式")]
    public Color outlineColor = Color.white;
    [Range(0.0f, 1.0f)]
    public float outlineIntensity = 1.0f;    // 轮廓强度（叠加透明度）

    [Header("可选：显示原始图 or 只显示边缘")]
    public bool showOriginal = true;          // true = 原图+轮廓, false = 纯黑底+轮廓
    public RenderTexture privateRT;
    private Material postEffectMat;

    
    [SerializeField] private Shader edgeDetectShader;

    void OnEnable()
    {
        if (edgeDetectShader == null)
            edgeDetectShader = Shader.Find("Hidden/EdgeDetectionOutline");
        if (edgeDetectShader != null && postEffectMat == null)
            postEffectMat = new Material(edgeDetectShader);
    }

  
    void OnRenderImage(RenderTexture src, RenderTexture dest)
    {
        // 1. 正常显示原始画面
        Graphics.Blit(src, dest);
        // 2. 私下处理轮廓并保存（不显示）
        if (postEffectMat != null)
        {
            postEffectMat.SetFloat("_EdgeThreshold", edgeThreshold);
            postEffectMat.SetColor("_OutlineColor", outlineColor);
            postEffectMat.SetFloat("_OutlineIntensity", outlineIntensity);
            postEffectMat.SetFloat("_ShowOriginal", showOriginal ? 1.0f : 0.0f);
            Graphics.Blit(src, privateRT, postEffectMat);
          
           
        }
    }
}