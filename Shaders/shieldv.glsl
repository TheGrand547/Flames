#version 440 core

#include "CubeMapMath"

layout(location = 0) in vec3 vPos;
layout(location = 1) in vec3 vNorm;
layout(location = 2) in vec2 vTex;

layout(location = 0) out vec4 fPos;
layout(location = 1) out vec4 fNorm;
layout(location = 2) out vec2 fTex;

uniform mat4 modelMat;
uniform mat4 normalMat;

layout(std140) uniform Camera
{
	mat4 View;
	mat4 Projection;
};
uniform int FeatureToggle;
uniform sampler2D textureIn;


void main()
{
	fNorm = normalMat * vec4(vNorm, 0);
	fPos = modelMat * vec4(vPos, 1.0);
	
	if (FeatureToggle > 0)
	{
		gl_Position = Projection * View * modelMat * vec4(vPos, 1.0);
	}
	else
	{
		vec3 offset = vNorm * (texture(textureIn, NormToUVCubemap(vNorm)).r - 0.1f) * 0.05f;
		gl_Position = Projection * View * modelMat * vec4(vPos + offset, 1.0);
	}
	fTex = vTex;
}