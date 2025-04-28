#version 440 core
#include "lighting"
#include "camera"

layout(location = 0) in vec2 textureCoords;
layout(location = 0) out vec4 fColor;

layout(location = 0) uniform sampler2D position;
layout(location = 1) uniform sampler2D normal;
layout(location = 2) uniform sampler2D color;


uniform vec3 lightPos;
uniform vec3 lightDir;

void main()
{	
	float ambient = 0.05f;
	
	vec3 tempLightColor = vec3(1, 0, 0);
	
	vec4 rawPos     = texture(position, textureCoords);
	vec3 fPos       = rawPos.xyz;
	vec3 normal     = texture(normal, textureCoords).xyz;
	vec3 shapeColor = texture(color, textureCoords).xyz;
	
	vec3 viewDirection = normalize(View[3].xyz - fPos);
	
	vec3 lightOut = DirectedPointLight(lightPos, -lightDir, tempLightColor, normal, fPos, viewDirection);
	
	for (int i = 0; i < 12; i++)
	{
		lightOut += PointLight(lightBuffer[i].position.xyz, lightBuffer[i].color.xyz, normal, fPos, viewDirection);
	
	}
	vec3 result = ambient * shapeColor + lightOut;
	fColor = vec4(result, 1.0);
	gl_FragDepth = rawPos.w;
}