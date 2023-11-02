#version 440 core

layout(location = 0) in vec2 fTex;

out vec4 color;

uniform sampler2D fontTexture;

void main()
{
	float value = texture(fontTexture, fTex).r;
	if (value == 0)
	{
		discard;
	}
	color = vec4(1, 1, 1, 1) * value;
}