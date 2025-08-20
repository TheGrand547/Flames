#version 440 core

layout(location = 0) out vec4 colorOut;

layout(location = 0) in vec4 fPos;
layout(location = 1) in vec4 fNorm;
layout(location = 2) in vec2 fTex;

uniform sampler2D textureIn;

void main()
{
	colorOut = vec4(120,204,226, 255) / 255 * sqrt(texture(textureIn, fTex).r);
}