#version 440 core

layout(location = 0) out vec4 colorOut;
layout(location = 1) out vec4 normalOut;

uniform vec4 Color;

void main()
{
	colorOut = Color;
	normalOut = Color;
}