#version 440 core

layout(location = 0) out vec4 colorOut;

uniform vec4 Color;

void main()
{
	colorOut = Color;
}