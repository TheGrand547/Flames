#version 440 core

flat in vec4 fNorm;
in vec3 fPos;

layout(location = 0) out vec4 fColor;
layout(location = 1) out vec4 fNormOut;

uniform vec3 lightColor;
uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 shapeColor;

void main()
{
	fNormOut = abs(fNorm);

	float ambient = 0.2f; // TODO: material setting
	vec3 ambientColor = lightColor * ambient;
	
	vec3 norm = fNorm.xyz;
	vec3 lightDir = normalize(lightPos - fPos);
	
	float diffuse = max(dot(norm, lightDir), 0.0);
	vec3 diffuseColor = diffuse * lightColor;
	
	vec3 viewDirection = normalize(viewPos - fPos);
	vec3 reflected = reflect(-lightDir, norm);

	float specular = pow(max(dot(viewDirection, reflected), 0.0), 128); // TODO: Specular setting
	vec3 specularOut = lightColor * specular; // TODO: I don't remember

	vec3 result = shapeColor * (ambientColor + diffuseColor + specularOut);
	fColor = vec4(result, 1.0);
}