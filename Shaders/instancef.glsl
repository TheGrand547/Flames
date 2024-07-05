#version 440 core

layout(location = 0) in vec3 fPos;
layout(location = 1) in vec2 fTex;
layout(location = 2) in vec3 transformedLightPos;
layout(location = 3) in vec3 transformedViewPos;
layout(location = 4) in vec3 transformedFPos;

layout(location = 0) out vec4 colorOut;
layout(location = 1) out vec4 normalOut;

uniform vec3 lightColor;
uniform sampler2D textureIn;
uniform sampler2D normalMapIn;
uniform sampler2D depthMapIn;
uniform sampler2D ditherMap;

uniform int newToggle;

void main()
{
	// TODO: something is kinda wrong about this and idk what
	float ambient = 0.2f; // TODO: material setting
	
	vec3 ambientColor = lightColor * ambient;
	vec3 viewDirection, norm;
	vec3 lightDir;// = normalize(lightPos - fPos);
	vec2 samplePoint = fTex;
	// Have to define lightDir, norm, and viewDirection
	
		
	viewDirection = normalize(transformedViewPos - transformedFPos);
	if (newToggle > 0)
	{
		float depth = texture(depthMapIn, fTex).r;
		samplePoint = fTex - viewDirection.xy * (depth * 0.1f);
		//if (samplePoint.x > 1.0 || samplePoint.x < 0.0 || samplePoint.y > 1.0 || samplePoint.y < 0.0)
			//discard;
	}
	norm = texture(normalMapIn, samplePoint).rgb;
	norm = 2 * norm - 1;
	lightDir = normalize(transformedLightPos - transformedFPos);
	
	float diffuse = max(dot(norm, lightDir), 0.0);
	vec3 diffuseColor = diffuse * lightColor;
	
	vec3 reflected = reflect(-lightDir, norm);

	float specular = pow(max(dot(viewDirection, reflected), 0.0), 128); // TODO: Specular setting
	vec3 specularOut = lightColor * specular;

	vec3 color = vec3(texture(textureIn, samplePoint));
	
	vec3 result = color * (ambientColor + diffuseColor + specularOut);
	
	normalOut = vec4(norm / 2.0 + 0.5, 1);
	
	// Dither stuff
	float dither = texture(ditherMap, gl_FragCoord.xy / 16).r;
	result.rgb += vec3(1, 1, 1) * mix(-0.5 / 255, 0.5 / 255, dither);
	colorOut = vec4(result, 1.0);

	//colorOut = vec4(abs(vec3(normalize(gl_FragCoord.xy), fTex.x * fTex.y) * viewDirection), 1.0);
	//colorOut = vec4(1, 0, 0, 1);
}