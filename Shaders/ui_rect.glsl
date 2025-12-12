#version 440 core

#include "ScreenSpace"

#ifndef INSTANCED
uniform vec4 rectangle;
uniform vec4 color;
#endif 

#ifdef VERTEX

#ifdef INSTANCED
layout(location = 0) in vec4 rectangle;
layout(location = 1) in vec4 rectColor;
layout(location = 0) flat out vec4 color;

#endif // INSTANCED

void main()
{

	vec2 topLeft = rectangle.xy;
	vec2 delta = rectangle.zw;
	
	uint id = gl_VertexID % 4;
	
	vec2 pos = topLeft;
	if (id % 2 == 1)
	{
		pos += vec2(0, delta.y);
	}
	if (id > 1)
	{
		pos += vec2(delta.x, 0);
	}
	// 1 is right on top of the camera
	gl_Position = Projection * vec4(pos.xy, 0, 1);
#ifdef INSTANCED
	color = rectColor;
#endif // INSTANCED
}

#endif // VERTEX

#ifdef FRAGMENT

#ifdef INSTANCED
layout(location = 0) flat in vec4 color;
#endif // INSTANCED

layout(location = 0) out vec4 fColor;

void main()
{
	fColor = color;
}

#endif // FRAGMENT