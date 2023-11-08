#version 440 core

layout(location = 0) in vec3 fPos;
layout(location = 1) in vec3 fNorm;
layout(location = 2) in vec2 fTex;

layout(location = 0) out vec4 colorOut;
layout(location = 1) out vec4 normalOut;

uniform vec3 lightColor;
uniform vec3 lightPos;
uniform vec3 viewPos;
uniform sampler2D textureIn;
uniform sampler2D ditherMap;

void main()
{
	float ambient = 0.2f; // TODO: material setting
	
	// TODO: light settings
	float constant = 1.0f;
	float linear = 0;
	float quadratic = 0;
	
	vec3 ambientColor = lightColor * ambient;
	
	vec3 norm = fNorm;
	vec3 lightDir = normalize(lightPos - fPos);
	
	float diffuse = max(dot(norm, lightDir), 0.0);
	vec3 diffuseColor = diffuse * lightColor;
		
	vec3 viewDirection = normalize(viewPos - fPos);
	vec3 reflected = reflect(-lightDir, norm);

	float specular = pow(max(dot(viewDirection, reflected), 0.0), 128); // TODO: Specular setting
	vec3 specularOut = lightColor * specular; // TODO: I don't remember

	float distance = length(lightPos - fPos);
	float attenuation = 1.0 / (constant + linear * distance + quadratic * (distance * distance));    

	vec3 color = vec3(texture(textureIn, fTex));
	
	vec3 result = color * (ambientColor + diffuseColor + specularOut) * attenuation;
	
	normalOut = vec4(abs(norm), 1);
	
	// Dither stuff
	
	float dither = texture(ditherMap, gl_FragCoord.xy / 16).r;
	result.rgb += vec3(1, 1, 1) * mix(-0.5 / 255, 0.5 / 255, dither);
	colorOut = vec4(result, 1.0);
	//colorOut = vec4(abs(vec3(normalize(gl_FragCoord.xy), fTex.x * fTex.y) * viewDirection), 1.0);
}