#version 440 core

#include "ScreenSpace"

#ifdef VERTEX

layout(location = 0) in vec2 vPos;
layout(location = 1) in vec2 vTex;

layout(location = 0) out vec2 fTex;

void main()
{
	gl_Position = Projection * vec4(vPos, 0, 1);
	fTex = vTex;
}

#endif // VERTEX

#ifdef FRAGMENT

layout(location = 0) in vec2 fTex;

layout(location = 0) out vec4 color;

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
#endif // FRAGMENT