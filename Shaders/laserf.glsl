#version 440 core
layout(location = 0) in float distance;

layout(location = 0) out vec4 colorOut;

layout(location = 0) uniform vec4 Color;

void main()
{
	colorOut = Color;
	colorOut.w *= max(distance, 0.5);
}