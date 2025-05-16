#version 440 core

#include "camera"

layout(location = 0) in mat4 modelMat;
layout(location = 4) in mat4 normalMat;
layout(location = 8) in vec2 multiplier;

layout(location = 0) out vec3 fPos;
layout(location = 1) out vec2 fTex;
layout(location = 2) out mat3 TBNmat;

vec2 positions[] = {
	vec2( 1.0f, -1.0f), vec2(-1.0f, -1.0f), vec2(-1.0f,  1.0f),
	vec2(-1.0f,  1.0f), vec2( 1.0f,  1.0f), vec2( 1.0f, -1.0f)
};

void main()
{
	int modulo = gl_VertexID % 6;
	int otherMod = gl_VertexID % 3;
	vec2 adjusted = positions[modulo].xy;
	if (otherMod == 0)
	{
		adjusted -= adjusted * 2 * multiplier;
	}
	vec4 modelPos = modelMat * vec4(adjusted.x, 0, adjusted.y, 1);
	fPos = modelPos.xyz;
	gl_Position = Projection * View * modelPos;
	
	
	vec3 tangent = normalize(mat3(normalMat) * vec3(1.f, 0.f, 0.f));
	vec3 normal = mat3(normalMat) * vec3(0.f, 1.f, 0.f);
	tangent = normalize(tangent - normal * dot(normal, tangent));
	vec3 biTangent = normalize(cross(normal, tangent));
	TBNmat = mat3(tangent, biTangent, normal);
}