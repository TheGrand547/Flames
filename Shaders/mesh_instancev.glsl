#version 440 core
#include "camera"

layout(location = 0) in vec3 vPos;
layout(location = 1) in vec3 vNorm;
layout(location = 2) in vec3 vTan;
layout(location = 3) in vec3 vBtn;
layout(location = 4) in vec2 vTex;
layout(location = 5) in mat4 modelMat;
layout(location = 9) in mat4 normalMat;

layout(location = 0) out vec3 fPos;
layout(location = 1) out vec2 fTex;
layout(location = 2) out vec3 transformedLightPos;
layout(location = 3) out vec3 transformedViewPos;
layout(location = 4) out vec3 transformedFPos;

layout(std140) uniform Lighting
{
	vec4 lightColor;
	vec4 lightDirection;
};


void main()
{
	mat3 shifted = mat3(normalMat);
	fPos = vec3(modelMat * vec4(vPos, 1.0));
	gl_Position = Projection * View * modelMat * vec4(vPos, 1.0);
	fTex = vTex + vBtn.xy;
	
	vec3 tangent = normalize(shifted * vTan);
	vec3 normal = vNorm;
	tangent = normalize(tangent - normal * dot(normal, tangent));
	vec3 biTangent = normalize(cross(normal, tangent));
	
	mat3 TBN = transpose(mat3(tangent, biTangent, normal));
	
	vec3 lightPos = fPos + 5 * lightDirection.xyz;
	transformedLightPos = TBN * lightPos;
	transformedViewPos = TBN * View[3].xyz;
	transformedFPos = TBN * fPos;
}