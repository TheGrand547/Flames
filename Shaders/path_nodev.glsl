#version 440 core

layout(location = 0) in vec3 vPos;
layout(location = 1) in vec3 Position;

layout(std140) uniform Camera
{
	mat4 View;
	mat4 Projection;
};

void main()
{
	gl_Position = Projection * View * vec4(vPos + Position, 1.0);
}