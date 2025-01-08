#version 440 core

layout(location = 0) in vec3 colorOut;

layout(location = 0) out vec4 fColor;

void main()
{
	fColor = vec4(colorOut, 1.0);
}