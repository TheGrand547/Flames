#version 440 core

#include "camera"

layout(location = 0) in vec3 vPos;
layout(location = 1) in mat4 Model;

layout(location = 0) out vec4 fColor;

vec4 colors[] = {
	vec4(1.f, 0.f, 0.f, 1.f), 
	vec4(0.f, 1.f, 0.f, 1.f),
	vec4(0.f, 0.f, 1.f, 1.f)
};

void main()
{
	gl_Position = Projection * View * Model * vec4(vPos, 1.0);
	fColor = colors[int(gl_InstanceID) % 3];
}