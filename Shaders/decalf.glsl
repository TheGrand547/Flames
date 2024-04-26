#version 440 core

layout(location = 0) in vec2 fTex;

layout(location = 0) out vec4 colorOut;
layout(location = 1) out vec4 normalOut;

uniform sampler2D textureIn;

void main()
{
	colorOut = texture(textureIn, fTex);
	normalOut = colorOut;
}
