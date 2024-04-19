#version 440 core

layout(location = 0) in vec4 fColor;

layout(location = 0) out vec4 colorOut;
layout(location = 1) out vec4 normalOut;

void main()
{
	colorOut = fColor;
	normalOut = fColor;
}