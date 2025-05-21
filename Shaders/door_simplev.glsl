#version 440 core

#include "camera"

layout(location = 0) in mat4 modelMat;
layout(location = 4) in mat4 normalMat;
layout(location = 8) in vec2 multiplier;

layout(location = 0) out vec3 fPos;
layout(location = 1) out vec2 fTex;
layout(location = 2) out mat3 TBNmat;

vec2 positions[] = {
// (1, 0), (0, 0), (0, 1)
	vec2( 1.0f, -1.0f), vec2(-1.0f, -1.0f), vec2(-1.0f,  1.0f),
// (0, 1), (1, 1), (1, 0)
	vec2(-1.0f,  1.0f), vec2( 1.0f,  1.0f), vec2( 1.0f, -1.0f)
};

// TODO: switch between door type
uniform int type;

void main()
{
	int modulo = gl_VertexID % 6;
	int otherMod = gl_VertexID % 3;
	int halves = int(floor(modulo / 2)) * 3;
	vec2 adjusted = positions[modulo].xy;

	float signFlip = (modulo > 2) ? -1.f : 1.f;

	// First tri moves in the -x direction when opening, second tri +x
	fTex = (adjusted + 1) / 2;
	//Proper 'receding triangles' door
	if (otherMod == 0)
	{
		adjusted.x = mix(signFlip, -signFlip, multiplier.x);
		//fTex = (adjusted + 1) / 2;
		if (modulo <= 2)
		{
			fTex.x = 1;
		}
		else
		{
			//fTex.x = 1 - multiplier.x;
			//fTex.y = 1 - multiplier.x;
		}
		//fTex.x = 1 - multiplier.x;
		//fTex.y = multiplier.x;
	}
	else if (otherMod == 1)
	{
		//fTex.x = multiplier.x;
		//fTex.y = (mix(signFlip, -signFlip, multiplier.x) + 1) / 2;
		if (modulo <= 2)
		{
			fTex.x = multiplier.x;
		}
		else
		{
			fTex.x = 1 - multiplier.x;
		}
	}	
	else if (otherMod == 2)
	{
		adjusted.y = mix(signFlip, -signFlip, multiplier.x);
		//fTex = (adjusted + 1) / 2;
		if (modulo <= 2)
		{
			fTex.x = multiplier.x;
			fTex.y = 1 - multiplier.x;
		}
		else
		{
			fTex.y = multiplier.x;
			fTex.x = 1 - multiplier.x;
			//fTex.y = 1;
			//fTex.y = 1 - multiplier.x;
		}
	}


	/*
	// Square receding door
	if (adjusted.x == 1.f)
	{
		adjusted.x -= adjusted.x * 2 * multiplier.x;
	}
	*/
	
	vec4 modelPos = modelMat * vec4(adjusted.x, 0, adjusted.y, 1);
	fPos = modelPos.xyz;
	gl_Position = Projection * View * modelPos;
	
	
	vec3 tangent = normalize(mat3(normalMat) * vec3(1.f, 0.f, 0.f));
	vec3 normal = mat3(normalMat) * vec3(0.f, 1.f, 0.f);
	tangent = normalize(tangent - normal * dot(normal, tangent));
	vec3 biTangent = normalize(cross(normal, tangent));
	TBNmat = mat3(tangent, biTangent, normal);
}