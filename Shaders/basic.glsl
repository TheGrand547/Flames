#version 440 core
#include "camera"

#ifdef VERTEX

layout(location = 0) in vec3 vPos;

uniform mat4 Model;

void main()
{
	gl_Position = Projection * View * Model * vec4(vPos, 1.0);
}

#endif // VERTEX

#ifdef FRAGMENT
layout(location = 0) out vec4 colorOut;

uniform vec4 Color;

void main()
{
	colorOut = Color;
}

#endif // FRAGMENT