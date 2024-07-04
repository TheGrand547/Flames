#version 440 core

layout(location = 0) in vec3 fPos;
layout(location = 1) in vec2 fTex;
layout(location = 2) in vec3 fNorm;

layout(location = 0) out vec4 colorOut;
layout(location = 1) out vec4 normalOut;

uniform vec3 lightColor;
uniform sampler2D textureIn;

uniform int newToggle;

uniform vec3 lightPos;
uniform vec3 viewPos;

void main()
{
	// TODO: something is kinda wrong about this and idk what
	float ambient = 0.2f; // TODO: material setting
	
	vec3 ambientColor = lightColor * ambient;
		
	vec3 viewDirection = normalize(viewPos - lightPos);
	vec3 lightDir = normalize(lightPos - fPos);
	
	float diffuse = max(dot(fNorm, lightDir), 0.0);
	vec3 diffuseColor = diffuse * lightColor;
	
	vec3 reflected = reflect(-lightDir, fNorm);

	float specular = pow(max(dot(viewDirection, reflected), 0.0), 128); // TODO: Specular setting
	vec3 specularOut = lightColor * specular;

	vec3 color = vec3(texture(textureIn, fTex));
	vec3 result = color * (ambientColor + diffuseColor + specularOut);
	colorOut = vec4(color, 1);
	normalOut = vec4(abs(fNorm), 1);
}