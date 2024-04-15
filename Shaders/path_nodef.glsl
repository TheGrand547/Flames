#version 440 core

layout (location = 0) vec4 colorOut;
layout (location = 1) vec4 normalOut;

uniform vec4 color;

void main()
{
	colorOut = color;
	normalOut = color;
}