#version 440 core

layout(location = 0) in vec3 fTex;
layout(location = 0) out vec4 colorOut;

uniform samplerCube skyBox;

void main()
{
	colorOut = texture(skyBox, fTex);
}