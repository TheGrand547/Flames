#version 440 core

layout(location = 0) in vec3 vPos;
layout(location = 1) in vec2 vTex;
layout(location = 2) in vec3 vTan;
layout(location = 3) in vec3 vBtn;
layout(location = 4) in mat4 Model;

layout(location = 0) out vec3 fPos;
layout(location = 1) out vec2 fTex;
layout(location = 2) out vec3 transformedLightPos;
layout(location = 3) out vec3 transformedViewPos;
layout(location = 4) out vec3 transformedFPos;

layout(std140) uniform Camera
{
	mat4 View;
	mat4 Projection;
};

uniform vec3 lightPos;
uniform vec3 viewPos;

void main()
{
	mat3 normalMat = mat3(Model); //mat3(transpose(inverse(Model)));
	// TODO: Normal
	fNorm = normalize(normalMat * vec3(0, 1, 0));
	fPos = vec3(Model * vec4(vPos, 1.0));
	gl_Position = Projection * View * Model * vec4(vPos, 1.0);
	fTex = vTex;
	
	vec3 tangent = normalize(normalMat * vTan);
	vec3 normal = fNorm;
	tangent = normalize(tangent - normal * dot(normal, tangent));
	vec3 biTangent = cross(normal, tangent);
	
	mat3 TBN = transpose(mat3(tangent, biTangent, normal));
	
	transformedLightPos = TBN * lightPos;
	transformedViewPos = TBN * viewPos;
	transformedFPos = TBN * fPos;
}