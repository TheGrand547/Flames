#version 440 core
#include "lighting"
#include "camera"

layout(location = 0) in vec2 textureCoords;
layout(location = 0) out vec4 fColor;

layout(location = 0) uniform sampler2D gPosition;
layout(location = 1) uniform sampler2D gNormal;
layout(location = 2) uniform sampler2D gColor;
layout(location = 3) uniform sampler2D gLighting;
layout(location = 4) uniform sampler2D gDepth;

uniform vec3 lightPos;
uniform vec3 lightDir;

void main()
{	
	float ambient = 0.05f;
	
	vec3 tempLightColor = vec3(1, 1, 1);
	
	vec4 rawPos     = texture(gPosition, textureCoords);
	vec3 normal     = texture(gNormal, textureCoords).xyz;
	vec3 shapeColor = texture(gColor, textureCoords).xyz;
	vec3 lightOut   = texture(gLighting, textureCoords).xyz;
	gl_FragDepth    = texture(gDepth, textureCoords).r;
	
	vec3 fragmentPosition = rawPos.xyz;
	vec3 viewDirection = normalize(View[3].xyz - fragmentPosition);
	lightOut += DirectedPointLight(lightPos, -lightDir, tempLightColor, normal, fragmentPosition, viewDirection);
	
	vec3 result = ambient * shapeColor + lightOut;
	fColor = vec4(result, 1.0);
}