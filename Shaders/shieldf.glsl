#version 440 core
#include "CubeMapMath"

layout(location = 0) out vec4 colorOut;

layout(location = 0) in vec4 fPos;
layout(location = 1) in vec4 fNorm;
layout(location = 2) in vec2 fTex;

uniform sampler2D textureIn;
uniform int FeatureToggle;

void main()
{
	const vec4 ShieldColor = vec4(120,204,226, 255) / 255;
	colorOut = ShieldColor * texture(textureIn, NormToUVCubemap(fNorm.xyz)).r;
	colorOut.w *= colorOut.w * colorOut.w;
}