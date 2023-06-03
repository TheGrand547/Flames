#version 440 core

layout(location = 0) out vec4 colorOut;

uniform vec3 color;

void main()
{
	colorOut = vec4(color, 1);
}