#version 440 core

layout(location = 0) in vec2 fTex;

out vec4 color;

uniform sampler2D fontTexture;

void main()
{
	color = vec4(1, 1, 1, 1) * texture(fontTexture, fTex).r;
	if (color.r == 0)
	{
		discard;
	}
}