#version 440 core

in vec3 vPos;
in vec2 vTex;

out vec3 fNorm;
out vec3 fPos;
out vec2 fTex;

uniform mat4 model;
uniform mat4 vp; 
// TODO: Normal matrix

void main()
{
	// TODO: Add normal as a vertex attribute and have the rotation/translation matrix as a separate parameter?
	//Normal = mat3(transpose(inverse(model))) * aNormal;  
	//normal = normalize(vec3(model * vec4(0, 1, 0, 0)));
	fNorm = normalize(mat3(transpose(inverse(model))) * vec3(0, 1, 0));
	fPos = vec3(model * vec4(vPos, 1.0));
	gl_Position = vp * model * vec4(vPos, 1.0);
	fTex = vTex;
}