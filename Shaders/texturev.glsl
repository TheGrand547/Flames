#version 440 core

layout(location = 0) in vec3 vPos;
layout(location = 1) in vec2 vTex;

layout(location = 0) out vec2 fTex;

layout(std140) uniform Camera
{
	mat4 View;
	mat4 Projection;
};

uniform mat4 orient;

void main()
{
	gl_Position = Projection * View * orient * vec4(vPos, 1.0);
	fTex = vTex;
}