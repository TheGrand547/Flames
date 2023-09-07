#version 440 core

layout(location = 0) in vec3 vPos;
layout(location = 1) in vec3 vNorm;
layout(location = 2) in vec3 vColor;
layout(location = 3) in vec2 vTex;

out vec4 fInNorm;
out vec4 fInColor;
out vec3 fInPos;

uniform mat4 modelMat;
uniform mat4 normMat;
uniform mat4 viewProjMat; 

void main()
{
	fInNorm = normMat * vec4(vNorm, 0);
	fInPos = vec3(modelMat * vec4(vPos, 1.0));
	gl_Position = viewProjMat * modelMat * vec4(vPos, 1.0);
	fInColor = vColor;
}