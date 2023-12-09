#version 440 core

layout(location = 0) in vec3 vPos;
layout(location = 1) in vec2 vTex;

layout(location = 0) out vec3 fPos;
layout(location = 1) out vec3 fNorm;
layout(location = 2) out vec2 fTex;

uniform mat4 Model;
// TODO: Normal matrix

layout(std140) uniform Camera
{
	mat4 View;
	mat4 Projection;
};

void main()
{
	fNorm = normalize(mat3(transpose(inverse(Model))) * vec3(0, 1, 0));
	fPos = vec3(Model * vec4(vPos, 1.0));
	//gl_Position = Projection * View * Model * vec4(vPos, 1.0);
	gl_Position = Model * vec4(vPos, 1.0);
	fTex = vTex;
}