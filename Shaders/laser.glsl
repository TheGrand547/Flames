#version 440 core
#include "camera"

#ifdef VERTEX

layout(location = 0) in vec3 vPos;
layout(location = 0) out float distance;

void main()
{
	gl_Position = Projection * View  * vec4(vPos, 1.0);
	distance = ((gl_VertexID % 2) == 0) ? 0.f : 1.f;
}

#endif // VERTEX

#ifdef FRAGMENT

layout(location = 0) in float distance;

layout(location = 0) out vec4 colorOut;

layout(location = 0) uniform vec4 Color;

void main()
{
	colorOut = Color;
	colorOut.w *= max(distance, 0.5);
}

#endif // FRAGMENT