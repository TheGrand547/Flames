#version 440 core
#include "camera"

layout(location = 0) in vec3 vPos;
layout(location = 0) out float distance;

void main()
{
	gl_Position = Projection * View  * vec4(vPos, 1.0);
	distance = ((gl_VertexID % 2) == 0) ? 0.f : 1.f;
}