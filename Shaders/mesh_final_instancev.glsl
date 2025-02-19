#version 440 core

layout(location = 0) in vec3 vPos;
layout(location = 1) in vec3 vNorm;
layout(location = 2) in vec2 vTex;
layout(location = 3) in mat4 modelMat;
layout(location = 7) in mat4 normalMat;

layout(location = 0) out vec3 colorOut;

layout(std140) uniform Camera
{
	mat4 View;
	mat4 Projection;
};

layout(std140) uniform Lighting
{
	vec4 lightColor;
	vec4 lightDirection;
};

uniform vec3 shapeColor;

void main()
{
	//vec3 fPos = (modelMat * vec4(vPos, 1.0)).xyz;
	vec3 norm = mat3(normalMat) * vNorm;
	
	gl_Position = Projection * View * modelMat * vec4(vPos, 1.0);

	float ambient = 0.25f;
	float diffuse = max(dot(norm, lightDirection.xyz), 0.0);
	colorOut = shapeColor * (ambient + diffuse)* lightColor.xyz;
}