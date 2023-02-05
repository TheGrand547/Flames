#version 440 core

in vec3 normal;
in vec3 fragPos;
in vec2 tex;
out vec4 fColor;

uniform vec3 lightColor;
uniform vec3 lightPos;
uniform vec3 viewPos;
uniform sampler2D textureIn;
uniform sampler2D ditherMap;
// TODO: normal/parallax mapping

void main()
{
	float ambient = 0.2f; // TODO: material setting
	vec3 ambientColor = lightColor * ambient;
	
	vec3 norm = normal;
	vec3 lightDir = normalize(lightPos - fragPos);
	
	float diffuse = max(dot(norm, lightDir), 0.0);
	vec3 diffuseColor = diffuse * lightColor;
	
	vec3 viewDirection = normalize(viewPos - fragPos);
	vec3 reflected = reflect(-lightDir, norm);

	float specular = pow(max(dot(viewDirection, reflected), 0.0), 128); // TODO: Specular setting
	vec3 specularOut = lightColor * specular; // TODO: I don't remember

	vec3 colorOut = vec3(texture(textureIn, tex));
	vec3 result = colorOut * (ambientColor + diffuseColor + specularOut);
	
	// Dither stuff
	
	float dither = texture(ditherMap, gl_FragCoord.xy / 160.).r;
	result.rgb += vec3(1, 1, 1) * mix(-0.5 / 255, 0.5 / 255, dither);
	
	fColor = vec4(result, 1.0);
}