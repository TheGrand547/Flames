#version 440 core

#include "ScreenSpace"

uniform sampler2D image;

#ifdef VERTEX

#ifdef INSTANCED
layout(location = 0) in vec4 rectangle;
#else // INSTANCED
uniform vec4 rectangle;
#endif // INSTANCED

layout(location = 0) out vec2 fTex;

vec2 textureLUT[] = {vec2(0, 1), vec2(0, 0), vec2(1, 1), vec2(1, 0)};

// TODO: Move to assuming position and stuff are right or something idk
void main()
{
	vec2 pos = rectangle.xy;
#ifdef SAFE
	vec2 size = min(textureSize(image, 0).xy, rectangle.zw);
#else // SAFE
	vec2 size = rectangle.zw;
#endif // SAFE
	pos += ceil((rectangle.zw - size) / 2);
	
	if ((gl_VertexID % 4) % 2 == 1)
	{
		pos += vec2(0, size.y);
	}
	if (gl_VertexID % 4 > 1)
	{
		pos += vec2(size.x, 0);
	}
	fTex = textureLUT[gl_VertexID % 4];
	gl_Position = Projection * vec4(pos.xy, 0, 1);
}

#endif // VERTEX

#ifdef FRAGMENT

layout(location = 0) in vec2 fTex;

layout(location = 0) out vec4 fColor;

void main()
{	
	fColor = texture(image, fTex);
	if (fColor.r == 0)
	{
		discard;
	}
}

#endif // FRAGMENT