#version 440 core

layout(location = 0) in vec3 vPos;
layout(location = 1) in vec4 instancePosition;
layout(location = 2) in vec3 color;
layout(location = 3) in vec3 constant;

layout(location = 0) out vec3 lightColor;
layout(location = 1) out vec3 lightCenter;
layout(location = 2) out float lightRadius;
layout(location = 3) out vec3 lightConstant;

#include "camera"

void main()
{
	float radius = instancePosition.w;
	lightColor  = color;
	lightCenter = instancePosition.xyz;
	lightRadius = radius;
	lightConstant = constant;
	gl_Position = Projection * View * vec4(vPos * radius + lightCenter, 1);
}