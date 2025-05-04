#version 440 core

layout(location = 0) in vec3 vPos;
layout(location = 1) in vec4 instancePosition;
layout(location = 2) in vec3 color;

layout(location = 0) out vec3 lightColor;
layout(location = 1) out vec3 lightCenter;
layout(location = 2) out float lightRadius;

#include "camera"

void main()
{
	float radius = instancePosition.w;
	lightColor  = color;
	lightCenter = instancePosition.xyz;
	lightRadius = radius;
	gl_Position = Projection * View * vec4(vPos * radius + lightCenter, 1);
}