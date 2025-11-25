#version 440 core
#include "CubeMapMath"
#include "camera"

layout(location = 0) out vec4 colorOut;

layout(location = 0) in vec4 fPos;
layout(location = 1) in vec4 fNorm;

uniform sampler2D textureIn;
uniform vec3 CameraPos;

void main()
{
	const vec4 ShieldColor = vec4(120,204,226, 255) / 255;
	colorOut = ShieldColor * texture(textureIn, NormToUVCubemap(fNorm.xyz)).r;
	
	const vec3 viewDirection = normalize(CameraPos - fPos.xyz);
	
	float alignment = 1.f - abs(dot(viewDirection, normalize(fNorm.xyz)));
	alignment = pow(alignment, 2);
	colorOut.w *= smoothstep(0.f, colorOut.w, alignment);

}