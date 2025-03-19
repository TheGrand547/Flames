#version 440 core

layout(location = 0) in vec3 vPos;
layout(location = 1) in vec3 vColor;
layout(location = 2) in mat4 modelMat;

layout(location = 0) out vec3 colorOut;

layout(std140) uniform Camera
{
	mat4 View;
	mat4 Projection;
};

void main()
{
	gl_Position = Projection * View * modelMat * vec4(vPos, 1.0);

	colorOut = vColor;
}