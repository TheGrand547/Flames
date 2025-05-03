#version 440 core

layout(location = 0) in vec3 vPos;
layout(location = 1) in vec3 vNorm;
layout(location = 2) in vec2 vTex;

layout(location = 0) out vec3 colorOut;

uniform mat4 modelMat;
uniform mat4 normalMat;

#include "camera"

layout(std140) uniform Lighting
{
	vec4 lightColor;
	vec4 lightDirection;
};

uniform vec3 shapeColor;

void main()
{
	vec3 norm = (normalMat * vec4(vNorm, 0)).xyz;
	
	gl_Position = Projection * View * modelMat * vec4(vPos, 1.0);

	float ambient = 0.45f;
	float diffuse = max(dot(norm, lightDirection.xyz), 0.0);
	colorOut = shapeColor * (ambient + diffuse)* lightColor.xyz;
}