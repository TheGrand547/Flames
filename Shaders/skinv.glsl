#version 440 core

layout(location = 0) in vec3 vPos;
layout(location = 1) in vec2 vTex;

layout(location = 0) out vec3 fPos;
layout(location = 1) out vec2 fTex;
layout(location = 2) out vec3 fNorm;

layout(std140) uniform Camera
{
	mat4 View;
	mat4 Projection;
};

uniform mat4 mats[2];

vec2 weights[] = 
{
	vec2(1, 0),
	vec2(1, 0),
	vec2(0.75, 0.25),
	
	vec2(0.75, 0.25),
	vec2(0.25, 0.75),
	
	vec2(0.25, 0.75),
	vec2(0, 1),
	vec2(0, 1),
};

void main()
{
	int index = gl_VertexID % 8;

	const int count = 2;
	vec4 temp = vec4(0);
	for (int i = 0; i < 2; i++)
	{
		temp += mats[i] * vec4(vPos, 1) * weights[index][i];
	}
	temp += vec4(0, 4, 0, 0);
	fPos = temp.xyz;
	gl_Position = Projection * View * temp;
	
	fTex = vTex;
	fNorm = vec3(0, 1, 0);
}