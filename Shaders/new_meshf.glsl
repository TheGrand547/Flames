#version 440 core

layout(location = 0) in vec3 fPos;
layout(location = 1) in vec2 fTex;
layout(location = 2) in vec3 transformedLightPos;
layout(location = 3) in vec3 transformedViewPos;
layout(location = 4) in vec3 transformedFPos;
layout(location = 5) in vec3 transformedLightDir;

layout(location = 0) out vec4 fColor;

uniform int featureToggle;

vec3 DirectedLight(vec3 lightDirection, vec3 lightColor, vec3 fragNormal, vec3 viewDirection)
{
	vec3 lightDir  = -lightDirection;
	float diffuse  = max(dot(fragNormal, lightDir), 0.0);
	vec3 reflected = reflect(-lightDir, fragNormal);
	float specular = pow(max(dot(viewDirection, reflected), 0.0), 128);
	return lightColor * (specular + diffuse);
}

vec3 PointLight(vec3 lightPos, vec3 lightColor, vec3 fragNormal, vec3 fragPos, vec3 viewDirection)
{
	vec3 lightDir  = normalize(lightPos - fragPos);
	float distance = length(lightPos - fragPos);
	float diffuse  = max(dot(fragNormal, lightDir), 0.0);
	vec3 reflected = reflect(-lightDir, fragNormal);
	float specular = pow(max(dot(viewDirection, reflected), 0.0), 128); 
	
	// Hacky but passable thing
	float fallOff = 1.0 / (1.0 + (1.0 / 10.0) * distance); 
	return lightColor * (specular + diffuse) * fallOff;
}

vec3 DirectedPointLight(vec3 lightPos, vec3 lightDirection, vec3 lightColor, vec3 fragNormal, vec3 fragPos, vec3 viewDirection)
{
	vec3 lightDir  = normalize(lightPos - fragPos);
	float distance = length(lightPos - fragPos);
	float diffuse  = max(dot(fragNormal, lightDir), 0.0);
	vec3 reflected = reflect(-lightDir, fragNormal);
	float specular = pow(max(dot(viewDirection, reflected), 0.0), 1000); // TODO: Specular setting
	
	// Check how aligned the "primary" direction of the light is, with the light hitting this fragment
	float directed = abs(dot(lightDir, lightDirection));
	
	float constant = cos(radians(25));
	// If the alignment is less than 25 degrees(arbitrary constant), the light has no effect
	float multiplier = step(constant, directed);
	multiplier *= (directed - constant) / ( 1 - constant);
	multiplier = pow(multiplier, 2);
	// Hacky but passable thing
	float fallOff = 1.0 / (1.0 + (1.0 / 50.0) * distance); 
	//fallOff = 1.0;
	return lightColor * (specular + diffuse) * fallOff * multiplier;
}

void main()
{
	float ambient = 0.2f;
	//vec3 norm = texture(normalMapIn, samplePoint).rgb;
	//norm = 2 * norm - 1;
	
	vec3 tempLightColor = vec3(1.0, 1.0, 1.0);
	vec3 tempShapeColor = vec3(1.0, 0.0, 0.0);
	
	vec3 norm = vec3(0, 0, 1);
	vec3 viewDirection = normalize(transformedViewPos - transformedFPos);
	
	vec3 lightOut;
	if (featureToggle == -1)
	{
		lightOut = PointLight(transformedLightPos, tempLightColor, norm, transformedFPos, viewDirection);
	}
	else
	{
		lightOut = DirectedPointLight(transformedLightPos, -transformedLightDir, tempLightColor, norm, transformedFPos, viewDirection);
	}
	
	vec3 result = ambient * tempShapeColor + lightOut;
	
	fColor = vec4(result, 1.0);
}