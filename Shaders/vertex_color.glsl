#version 440 core

#include "camera"

#ifdef VERTEX

layout(location = 0) in vec3 vPos;
layout(location = 1) in vec3 vColor;
layout(location = 2) in mat4 modelMat;

layout(location = 0) out vec3 colorOut;


void main()
{
	gl_Position = Projection * View * modelMat * vec4(vPos, 1.0);

	colorOut = vColor;
}

#endif // VERTEX

#ifdef FRAGMENT

layout(location = 0) in vec3 colorOut;
layout(location = 0) out vec4 fColor;

void main()
{
	fColor = vec4(colorOut, 1.f);
}

#endif // FRAGMENT