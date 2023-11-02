#version 440 core

layout(location = 0) in vec2 vPos;
layout(location = 1) in vec2 vTex;

layout(location = 0) out vec2 fTex;

uniform mat4 screenProjection;

void main()
{
	gl_Position = screenProjection * vec4(vPos, 0, 1);
	fTex = vTex;
}