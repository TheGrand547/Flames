#version 440 core

in vec3 vPos;
in vec2 vTex;

out vec3 fNorm;
out vec3 fPos;
out vec2 fTex;

uniform mat4 Model;
// TODO: Normal matrix

layout(std140) uniform Camera
{
	mat4 View;
	mat4 Projection;
};

void main()
{
	// TODO: Add normal as a vertex attribute and have the rotation/translation matrix as a separate parameter?
	//Normal = mat3(transpose(inverse(model))) * aNormal;  
	//normal = normalize(vec3(model * vec4(0, 1, 0, 0)));
	fNorm = normalize(mat3(transpose(inverse(Model))) * vec3(0, 1, 0));
	fPos = vec3(Model * vec4(vPos, 1.0));
	gl_Position = Projection * View * Model * vec4(vPos, 1.0);
	fTex = vTex;
}