#version 440 core

in vec3 normal;
in vec3 fragPos;
in vec2 tex;
layout(location = 0) out vec4 fColor;
layout(location = 1) out vec4 fNormal;

uniform vec3 lightColor;
uniform vec3 lightPos;
uniform vec3 viewPos;
uniform sampler2D textureIn;
uniform sampler2D ditherMap;
// TODO: normal/parallax mapping

void main()
{
	float ambient = 0.2f; // TODO: material setting
	
	// TODO: light settings
	float constant = 1.0f;
	float linear = 0;
	float quadratic = 0;
	
	vec3 ambientColor = lightColor * ambient;
	
	vec3 norm = normal;
	vec3 lightDir = normalize(lightPos - fragPos);
	
	float diffuse = max(dot(norm, lightDir), 0.0);
	vec3 diffuseColor = diffuse * lightColor;
		
	vec3 viewDirection = normalize(viewPos - fragPos);
	vec3 reflected = reflect(-lightDir, norm);

	float specular = pow(max(dot(viewDirection, reflected), 0.0), 128); // TODO: Specular setting
	vec3 specularOut = lightColor * specular; // TODO: I don't remember

	float distance = length(lightPos - fragPos);
	float attenuation = 1.0 / (constant + linear * distance + quadratic * (distance * distance));    

	vec3 colorOut = vec3(texture(textureIn, tex));
	
	
	vec3 result = colorOut * (ambientColor + diffuseColor + specularOut) * attenuation;
	
	fColor = vec4(result, 1);
	fNormal = vec4(abs(normal), 1);
	
	// Dither stuff
	
	float dither = texture(ditherMap, gl_FragCoord.xy / 16).r;
	const float maxVal = 255;
	vec3 scaled = result * maxVal;
	vec3 floored = floor(scaled);
	vec3 delta = scaled - floored;
		
	result = (floored + step(dither, delta)) / maxVal;
	
	fColor = vec4(result, 1.0);
}