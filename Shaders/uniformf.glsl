#version 440 core

layout(location = 0) out vec4 colorOut;

layout(location = 0) in vec4 fColor;

void main()
{
	colorOut = fColor;
}