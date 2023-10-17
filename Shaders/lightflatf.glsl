#version 440 core

in vec3 color;
in vec4 fNorm;
in vec3 fPos;

layout(location = 0) out vec4 fColor;
layout(location = 1) out vec4 fNormOut;

uniform vec3 lightColor;
uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 shapeColor;

uniform sampler2D hatching;

void main()
{
	// SAMPLE A TEXTURE BASED ON SCREEN POSITION AND TEXTURE COORDINATE PAIRS
	fNormOut = abs(fNorm);

	float ambient = 0.15f; // TODO: material setting
	
	vec3 norm = fNorm.xyz;
	vec3 lightDir = normalize(lightPos - fPos);
	
	float diffuse = max(dot(norm, lightDir), 0.0);
	// Quantized
	//vec3 diffuseColor = max(ambient, (ceil(diffuse * 6) - 1) / 5) * lightColor;
	
	vec3 viewDirection = normalize(viewPos - fPos);
	vec3 reflected = reflect(-lightDir, norm);

	float specular = pow(max(dot(viewDirection, reflected), 0.0), 6); // TODO: Specular setting

	vec3 result = shapeColor + (ambient + diffuse + specular) * lightColor;
	//result = (ceil(result * 5) - 1) / 4;
	fColor = vec4(result, 1.0);
}