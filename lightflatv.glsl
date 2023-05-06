#version 440 core

in vec3 vPos;
in vec3 vNorm;

out vec4 fNorm;
out vec3 fPos;

uniform mat4 modelMat;
uniform mat4 normMat;
uniform mat4 viewProjMat; 
// TODO: Normal matrix

void main()
{
	fNorm = normMat * vec4(vNorm, 0);
	fPos = vec3(modelMat * vec4(vPos, 1.0));
	gl_Position = viewProjMat * modelMat * vec4(vPos, 1.0);
}