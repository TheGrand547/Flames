#version 440 core

layout(location = 0) in vec3 vPos;
layout(location = 1) in vec3 vNorm;
layout(location = 2) in vec2 vTex;

out vec4 fPos;
out vec4 fNorm;
out vec2 fTex;

uniform mat4 modelMat;
uniform mat4 normalMat;

layout(std140) uniform Camera
{
	mat4 View;
	mat4 Projection;
};

void main()
{
	fNorm = normalMat * vec4(vNorm, 0);
	fPos = modelMat * vec4(vPos, 1.0);
	gl_Position = Projection * View * modelMat * vec4(vPos, 1.0);
	fTex = vTex;
}