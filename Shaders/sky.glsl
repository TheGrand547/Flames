#version 440 core

#ifdef VERTEX

layout(location = 0) in vec3 vPos;

layout(location = 0) out vec3 fTex;

#include "camera"

void main()
{
	fTex = vPos;
	
	vec4 temp = Projection * View * vec4(vPos, 0);
	gl_Position.xyw = temp.xyw;
	gl_Position.z = 0;
}

#endif // VERTEX

#ifdef FRAGMENT

layout(location = 0) in vec3 fTex;
layout(location = 0) out vec4 colorOut;

uniform samplerCube skyBox;

void main()
{
	colorOut = texture(skyBox, fTex);
}

#endif // FRAGMENT