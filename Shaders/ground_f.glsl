#version 440 core

layout(location = 0) in vec3 fPos;
layout(location = 1) in vec3 fNorm;
layout(location = 2) in vec2 fTex;

layout(location = 0) out vec4 colorOut;
layout(location = 1) out vec4 normalOut;

uniform int redLine;

void main()
{
	vec4 mult = vec4(1, 1, 1, 1);
	if (redLine != 0)
	{
		mult = vec4(1, 0, 0, 1);
	}
	colorOut = mult * fPos.y;
	colorOut.w = 1;
	normalOut = vec4(fNorm, 1);
}