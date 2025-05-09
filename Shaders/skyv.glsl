#version 440 core

layout(location = 0) in vec3 vPos;

layout(location = 0) out vec3 fTex;

#include "camera"

void main()
{
	fTex = vPos;
	
	vec4 temp = Projection * View * vec4(vPos, 0);
	gl_Position = temp.xyww;
}