#version 440 core

layout(location = 0) in vec3 vPos;
layout(location = 1) in vec2 vTex;
layout(location = 2) in vec3 vTan;
layout(location = 3) in vec3 vBtn;
layout(location = 4) in mat4 Model;

layout(location = 0) out vec3 fPos;
layout(location = 1) out vec3 fNorm;
layout(location = 2) out vec2 fTex;
layout(location = 3) out mat3 fTBN;

layout(std140) uniform Camera
{
	mat4 View;
	mat4 Projection;
};

void main()
{
	mat3 normalMat = mat3(Model); //mat3(transpose(inverse(Model)));
	// TODO: Normal
	fNorm = normalize(normalMat * vec3(0, 1, 0));
	fPos = vec3(Model * vec4(vPos, 1.0));
	gl_Position = Projection * View * Model * vec4(vPos, 1.0);
	fTex = vTex;
	
	vec3 tanget = normalize(normalMat * vTan);
	vec3 biTangent = normalize(normalMat * vBtn);
	
	fTBN = mat3(tanget, biTangent, fNorm);
}