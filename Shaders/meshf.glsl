#version 440 core

layout(location = 0) out vec4 colorOut;
layout(location = 1) out vec4 normalOut;


in vec4 fPos;
in vec4 fNorm;
in vec2 fTex;

uniform sampler2D textureIn;

void main()
{
	normalOut = abs(fNorm);
	colorOut = texture(textureIn, fTex);
}