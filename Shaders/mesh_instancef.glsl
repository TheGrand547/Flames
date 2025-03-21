#version 440 core

layout(location = 0) in vec3 fPos;
layout(location = 1) in vec2 fTex;
layout(location = 2) in vec3 transformedLightPos;
layout(location = 3) in vec3 transformedViewPos;
layout(location = 4) in vec3 transformedFPos;

layout(location = 0) out vec4 colorOut;


layout(std140) uniform Lighting
{
	vec4 lightColor;
	vec4 lightDirection;
};

uniform vec3 shapeColor;

uniform sampler2D normalMapIn;

void main()
{
	float ambient = 0.2f; // TODO: material setting
	
	vec3 ambientColor = lightColor.xyz * ambient;
	vec3 viewDirection, norm;
	vec3 lightDir;// = normalize(lightPos - fPos);
	vec2 samplePoint = fTex;
	// Have to define lightDir, norm, and viewDirection
	
	viewDirection = normalize(transformedViewPos - transformedFPos);
	
	norm = texture(normalMapIn, samplePoint).rgb;
	norm = 2 * norm - 1;
	//lightDir = normalize(lightDirection.xyz);
	lightDir = normalize(transformedLightPos - transformedFPos);
	
	float diffuse = max(dot(norm, lightDir), 0.0);
	vec3 diffuseColor = diffuse * lightColor.xyz;
	
	vec3 reflected = reflect(-lightDir, norm);

	float specular = pow(max(dot(viewDirection, reflected), 0.0), 128); // TODO: Specular setting
	vec3 specularOut = lightColor.xyz * specular;
	
	vec3 result = shapeColor * (ambientColor + diffuseColor + specularOut);
		
	colorOut = vec4(result, 1.0);
}