#version 440 core

layout(location = 0) in vec2 fTex;

out vec4 color;

uniform sampler2D fontTexture;

void main()
{
	color = texture(fontTexture, fTex);
}