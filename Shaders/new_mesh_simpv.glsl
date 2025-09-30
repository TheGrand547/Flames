#version 440 core

layout(location = 0) in vec3 vPos;
layout(location = 5) in mat4 modelMat;

#include "camera"

void main()
{
	gl_Position = Projection * View * modelMat * vec4(vPos, 1.0);
}