#version 440 core
#include "camera"

#ifdef VERTEX

layout(location = 0) in vec3 vPos;
layout(location = 1) in vec3 vColor;

layout(location = 0) out vec4 fColor;

uniform vec3 Color;

const int duration = 256;

void main()
{
	float ratio = float(gl_VertexID) / duration;
	vec3 local = vPos + (vColor * ratio * 1.25);

	gl_Position = Projection * View * vec4(local, 1.0);
	fColor.xyz = Color;
	fColor.w = 1 - pow(1 - ratio, 3);
}

#endif // VERTEX

#ifdef FRAGMENT

layout(location = 0) out vec4 colorOut;

layout(location = 0) in vec4 fColor;

void main()
{
	colorOut = fColor;
}

#endif // FRAGMENT