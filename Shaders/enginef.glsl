#version 440 core

layout(location = 0) in vec4 colorIn;

layout(location = 0) out vec4 colorOut;

void main()
{
	colorOut = colorIn;
}