#version 440 core

in vec4 fInColor;
in vec4 fInNorm;
in vec4 fInPos;

layout(location = 0) out vec4 fColor;
layout(location = 1) out vec4 fNorm;

void main()
{
	fColor = colorOut;
	fNorm =  normal;
}