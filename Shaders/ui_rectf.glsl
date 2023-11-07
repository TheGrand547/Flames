#version 440 core

layout(location = 0) out vec4 fColor;

uniform vec4 color;

void main()
{
	fColor = color;
}