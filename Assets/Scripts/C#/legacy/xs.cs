using System.Collections;
using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.UI;

public class xsScript : MonoBehaviour
{
    public RawImage raw;
    public Camera mainCamera;
    SpriteRenderer s;
    public GameObject g;
    public RenderTexture Texture;
    public Material cropMaterial;
    void Start()
    {
        g.transform.position = Vector3.zero;
        s =  g.transform.GetComponent<SpriteRenderer>();
        if (s == null) return;
        var b= s.sprite.bounds;
        raw.rectTransform.sizeDelta = new Vector2(b.max.x * 30, b.max.y * 30);
      
    }
    void UpdateCropRect()
    {
        if ( s == null || mainCamera == null) return;

        
        Vector3[] worldCorners = new Vector3[4];
        Bounds b = s.sprite.bounds;
        Vector3 localMin = b.min;
        Vector3 localMax = b.max;
        worldCorners[0] = s.transform.TransformPoint(new Vector3(localMin.x, localMin.y, 0));
        worldCorners[1] = s.transform.TransformPoint(new Vector3(localMax.x, localMin.y, 0));
        worldCorners[2] = s.transform.TransformPoint(new Vector3(localMax.x, localMax.y, 0));
        worldCorners[3] = s.transform.TransformPoint(new Vector3(localMin.x, localMax.y, 0));

        // 转为视口坐标（0~1）
        Vector2 minUV = Vector2.one;
        Vector2 maxUV = Vector2.zero;
        for (int i = 0; i < 4; i++)
        {
            Vector3 viewport = mainCamera.WorldToViewportPoint(worldCorners[i]);
            minUV = Vector2.Min(minUV, viewport);
            maxUV = Vector2.Max(maxUV, viewport);
        }

        // 设置材质参数
        if (cropMaterial != null)
        {
            cropMaterial.SetTexture("_MainTex", Texture);
            cropMaterial.SetVector("_RectMin", new Vector4(minUV.x, minUV.y, 0, 0));
            cropMaterial.SetVector("_RectMax", new Vector4(maxUV.x, maxUV.y, 0, 0));
        }
    }
    private void Update()
    { 
        UpdateCropRect();
    }
}
