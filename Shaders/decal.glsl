#version 440 core

#ifdef VERTEX
#include "camera"

layout(location = 0) in vec3 vPos;
layout(location = 1) in vec2 vTex;

layout(location = 0) out vec2 fTex;


void main()
{
	gl_Position = Projection * View * vec4(vPos, 1);
	fTex = vTex;
}

#endif // VERTEX

#ifdef FRAGMENT

layout(location = 0) in vec2 fTex;

layout(location = 0) out vec4 colorOut;
layout(location = 1) out vec4 normalOut;

uniform sampler2D textureIn;


void main()
{
	colorOut = texture(textureIn, fTex);
	normalOut = colorOut;
	gl_FragDepth = gl_FragCoord.z - 0.0001;
}

#endif // FRAGMENT