#version 440 core

in vec3 colorOut;
in vec3 normal;
in vec3 fragPos;
out vec4 fColor;

uniform vec3 lightColor;
uniform vec3 lightPos;

void main()
{
	float ambient = 0.9f;
	vec3 ambientColor = lightColor * ambient;
	
	vec3 norm = normal;
	vec3 lightDir = normalize(lightPos - fragPos);
	
	float diffuse = max(dot(norm, lightDir), 0.0);
	vec3 diffuseColor = diffuse * lightColor;
	
	fColor = vec4(colorOut * (ambientColor * diffuseColor), 1.0);
}