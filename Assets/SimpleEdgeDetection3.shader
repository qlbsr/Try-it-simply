Shader "Hidden/SimpleEdgeDetection"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _BinaryMode ("Binary Mode", Range(0,1)) = 1
        _EdgeColor ("Edge Color", Color) = (1,0,0,1)
        _Intensity ("Intensity", Range(0,1)) = 1
    }
    SubShader
    {
        Cull Off ZWrite Off ZTest Always

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            sampler2D _MainTex;
            float4 _MainTex_TexelSize;
            float _BinaryMode;
            float4 _EdgeColor;
            float _Intensity;

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            v2f vert (appdata_img v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.texcoord;
                return o;
            }

            // 判断一个像素是否为白色（前景）
            bool isWhite(float4 col, float mode)
            {
                if (mode > 0.5)  // 已经是二值图：直接取红色分量（或亮度>0.5）
                    return col.r > 0.5;
                else              // 普通彩色图：亮度 > 0.5 视为白
                {
                    float bright = dot(col.rgb, float3(0.299, 0.587, 0.114));
                    return bright > 0.5;
                }
            }

            fixed4 frag (v2f i) : SV_Target
            {
                float4 center = tex2D(_MainTex, i.uv);
                bool centerWhite = isWhite(center, _BinaryMode);

                // 如果中心不是白色，直接输出透明（或黑色背景）
                if (!centerWhite)
                    return fixed4(0,0,0,0);   // 完全透明

                // 检查4邻域（上下左右）
                float2 offsets[4] = {
                    float2( 0,  1),
                    float2( 0, -1),
                    float2( 1,  0),
                    float2(-1,  0)
                };

                bool isEdge = false;
                for (int i=0; i<4; i++)
                {
                    float2 neighborUV = i.uv + offsets[i] * _MainTex_TexelSize.xy;
                    float4 neighbor = tex2D(_MainTex, neighborUV);
                    if (!isWhite(neighbor, _BinaryMode))
                    {
                        isEdge = true;
                        break;
                    }
                }

                if (isEdge)
                {
                    // 边缘像素：输出指定的边缘颜色，强度可调
                    return fixed4(_EdgeColor.rgb, _EdgeColor.a * _Intensity);
                }
                else
                {
                    // 非边缘的白色像素（内部）：输出透明（或可改为其他颜色）
                    return fixed4(0,0,0,0);
                }
            }
            ENDCG
        }
    }
}