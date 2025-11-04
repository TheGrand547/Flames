#version 440 core

#include "lighting"
#include "camera"

layout(location = 0) in vec3 fPos;
layout(location = 1) in vec2 fTex;
layout(location = 2) in mat3 TBNmat;


layout(location = 0) out vec4 fColor;

uniform int featureToggle;

uniform vec3 lightPos;
uniform vec3 lightDir;

void main()
{
	float ambient = 0.05f;
	//vec3 norm = texture(normalMapIn, samplePoint).rgb;
	//norm = 2 * norm - 1;
	
	vec3 tempShapeColor = vec3(1.0, 1.0, 1.0);
	vec3 tempLightColor = vec3(1.0, 0.0, 0.0);
	
	vec3 norm = TBNmat * vec3(0, 0, 1);
	vec3 viewDirection = normalize(View[3].xyz - fPos);
	
	vec3 lightOut = DirectedPointLight(lightPos, -lightDir, tempLightColor, norm, fPos, viewDirection);
	vec3 result = ambient * tempShapeColor + lightOut;
	
	fColor = vec4(result, 1.0);
}