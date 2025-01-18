#version 440 core

layout(location = 0) in vec3 vPos;
layout(location = 1) in vec3 vNorm;
layout(location = 2) in vec2 vTex;

layout(location = 0) out vec3 colorOut;

uniform mat4 modelMat;
uniform mat4 normalMat;

layout(std140) uniform Camera
{
	mat4 View;
	mat4 Projection;
};

/*
layout(std140) uniform Lighting
{

};
*/

uniform vec3 lightColor;
uniform vec3 shapeColor;

void main()
{
	//vec3 fPos = (modelMat * vec4(vPos, 1.0)).xyz;
	vec3 norm = (normalMat * vec4(vNorm, 0)).xyz;
	
	gl_Position = Projection * View * modelMat * vec4(vPos, 1.0);

	float ambient = 0.25f;
	vec3 lightDir = normalize(vec3(0.15, 1, 0.15));
	float diffuse = max(dot(norm, lightDir), 0.0);
	colorOut = shapeColor * (ambient + diffuse)* lightColor;
}