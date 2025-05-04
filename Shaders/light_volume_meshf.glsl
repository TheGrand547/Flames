#version 440 core

layout(location = 0) in vec3 lightColor;
layout(location = 1) in vec3 lightCenter;
layout(location = 2) in float lightRadius;

layout(location = 0) out vec4 fColor;

#include "lighting"
#include "camera"

layout(location = 0) uniform sampler2D gPosition;
layout(location = 1) uniform sampler2D gNormal;

void main()
{
	// PAIN I HAVE LIVED IN ENDLESS PAIN NOW I PAY IT BACK IN SPADES
	// Needs to be in normalized device coordinates space
	vec2 textureCoords = gl_FragCoord.xy / 1000;
	
	vec4 rawPos     = texture(gPosition, textureCoords);
	vec3 fPos       = rawPos.xyz;
	vec3 normal     = texture(gNormal, textureCoords).xyz;
	
	vec3 viewDirection = normalize(View[3].xyz - fPos);
	vec3 lightOut = PointLight(lightCenter, lightColor, normal, fPos, viewDirection);
	fColor = vec4(lightOut, 1.0);
	//fColor = vec4(textureCoords, lightOut.z, 1.0);
}