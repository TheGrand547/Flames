#version 440 core

in vec4 vColor;

layout(location = 0) out vec4 fColor;
layout(location = 1) out vec4 fNormOut;


void main()
{	
	fColor = vColor;
	fNormOut = fColor;
}