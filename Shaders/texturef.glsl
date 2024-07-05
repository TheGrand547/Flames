#version 440 core

layout(location = 0) in vec2 fTex;

layout(location = 0) out vec4 fColor;

uniform sampler2D sampler;

void main()
{
	fColor = texture(sampler, fTex);
}