#version 440 core

layout(location = 0) in vec3 fPos;
layout(location = 1) in vec3 fNorm;
layout(location = 2) in vec2 fTex;
layout(location = 3) in mat3 fBTN;

layout(location = 0) out vec4 colorOut;
layout(location = 1) out vec4 normalOut;

uniform vec3 lightColor;
uniform vec3 lightPos;
uniform vec3 viewPos;
uniform sampler2D textureIn;
uniform sampler2D normalMapIn;
uniform sampler2D depthMapIn;
uniform sampler2D ditherMap;

uniform int flops;

void main()
{
	// TODO: something is kinda wrong about this and idk what
	float ambient = 0.2f; // TODO: material setting
	
	vec3 ambientColor = lightColor * ambient;
	vec3 viewDirection, norm;
	if (flops != 0)
	{
		mat3 invs = inverse(fBTN);
		
		viewDirection = normalize((invs * viewPos - invs * fPos));
		float factor = 0.1f;
		vec2 texLoc = (viewDirection.xy / viewDirection.z) * texture(depthMapIn, fTex).r * factor;
		
		if (flops != 0)
			texLoc = fTex - texLoc;
		else
			texLoc = fTex;
		//viewDirection = normalize(viewPos - fPos);
		
		norm = fNorm;
		vec3 inNorm = normalize(texture(normalMapIn, texLoc).xyz * 2.0 - 1.0);
		norm = fBTN * inNorm;
	}
	else
	{
		norm = fNorm;
		viewDirection = normalize(viewPos - fPos);
	}
	
	
	
	vec3 lightDir = normalize(lightPos - fPos);
	lightDir = vec3(0, 1, 0);
	
	float diffuse = max(dot(norm, lightDir), 0.0);
	vec3 diffuseColor = diffuse * lightColor;
	
	vec3 reflected = reflect(-lightDir, norm);

	float specular = pow(max(dot(viewDirection, reflected), 0.0), 128); // TODO: Specular setting
	vec3 specularOut = lightColor * specular;


	vec3 color = vec3(texture(textureIn, fTex));
	
	vec3 result = color * (ambientColor + diffuseColor + specularOut);
	
	normalOut = vec4(abs(norm), 1);
	
	// Dither stuff
	float dither = texture(ditherMap, gl_FragCoord.xy / 16).r;
	result.rgb += vec3(1, 1, 1) * mix(-0.5 / 255, 0.5 / 255, dither);
	colorOut = vec4(result, 1.0);

	//colorOut = vec4(abs(vec3(normalize(gl_FragCoord.xy), fTex.x * fTex.y) * viewDirection), 1.0);
	//colorOut = vec4(1, 0, 0, 1);
}