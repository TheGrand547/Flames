#version 440 core

layout(location = 0) in vec3 vPos;
layout(location = 1) in vec3 vNorm;
layout(location = 2) in vec3 vTan;
layout(location = 3) in vec3 vBtn;
layout(location = 4) in vec2 vTex;

#ifdef INSTANCED

layout(location = 5) in mat4 modelMat;
layout(location = 9) in mat4 normalMat;

#else // INSTANCED

uniform mat4 modelMat;
uniform mat4 normalMat;

#endif // INSTANCED

layout(location = 0) out vec3 fPos;
layout(location = 1) out vec2 fTex;
layout(location = 2) out mat3 TBNmat;

#include "camera"

void main()
{
	mat3 shifted = mat3(normalMat);
	fPos = vec3(modelMat * vec4(vPos, 1.0));
	gl_Position = Projection * View * modelMat * vec4(vPos, 1.0);
	fTex = vTex + vBtn.xy; // ?????
	
	vec3 tangent = normalize(shifted * vTan);
	vec3 normal  = normalize(shifted * vNorm);
	tangent = normalize(tangent - normal * dot(normal, tangent));
	vec3 biTangent = normalize(cross(normal, tangent));
	TBNmat = mat3(tangent, biTangent, normal);
}