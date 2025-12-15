#version 440 core

#include "camera"

#ifdef VERTEX

layout(location = 0) in vec3 vPos;
layout(location = 1) in vec3 vNorm;
layout(location = 2) in vec2 vTex;

#ifdef INSTANCED
layout(location = 3) in mat4 modelMat;
layout(location = 7) in mat4 normalMat;
#else
uniform mat4 modelMat;
uniform mat4 normalMat;
#endif // INSTANCED

layout(location = 0) out vec3 fPos;
layout(location = 1) out vec3 fNorm;
layout(location = 2) out vec2 fTex;

// TODO: gl_DrawID to switch textures <--- Investigate

void main()
{
	gl_Position = Projection * View * modelMat * vec4(vPos, 1.0);
	
	fPos  = (modelMat * vec4(vPos, 1.0)).xyz;
	fNorm = mat3(normalMat) * vNorm;
	fTex  = vTex;
}
#endif 


#ifdef FRAGMENT

#include "lighting"
#include "forward_buffers"
#include "frustums"
#include "cone"
#include "forward_plus"

layout(location = 0) in vec3 fPos;
layout(location = 1) in vec3 fNorm;
layout(location = 2) in vec2 fTex;

layout(location = 0) out vec4 fColor;

layout(location = 0) uniform sampler2D color;
layout(location = 1) uniform vec3 shapeColor;
layout(location = 2) uniform int checkUVs;
layout(location = 3) uniform vec3 CameraPos;

void main()
{
	vec3 viewDirection = normalize(CameraPos - fPos);
	FragData data;
	data.position = fPos;
	data.normal = fNorm;
	data.viewDirection = viewDirection;
	
	// TODO: Texturing(if desired)
	
	vec3 lightOut = ForwardPlusLighting(data);
	fColor = vec4(lightOut, 1.f);
}

#endif // FRAGMENT
