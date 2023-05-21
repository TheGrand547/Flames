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
	fNormOut = abs(fNorm);

	float ambient = 0.2f; // TODO: material setting
	vec3 ambientColor = lightColor * ambient;
	
	vec3 norm = fNorm.xyz;
	vec3 lightDir = normalize(lightPos - fPos);
	
	float diffuse = max(dot(norm, lightDir), 0.0);
	vec3 diffuseColor = diffuse * lightColor;
	
	vec3 viewDirection = normalize(viewPos - fPos);
	vec3 reflected = reflect(-lightDir, norm);

	float specular = pow(max(dot(viewDirection, reflected), 0.0), 4); // TODO: Specular setting
	vec3 specularOut = lightColor * specular; // TODO: I don't remember

	vec3 result = shapeColor * (ambientColor + diffuseColor);
	if (specular > 0.25)
	{
		//fNormOut = -fNormOut;
		result = vec3(1, 1, 1);
	}
	
	// TODO: specular thing
	float effect = 1 - (ambient + diffuse);
	vec4 hatch = texture(hatching, gl_FragCoord.xy / 1000);
	if (hatch.w == 0 && hatch.w != 0)
	{
		float hatchVal = 0;
		if (effect > 0.5)
		{
			hatchVal = hatch.r;
		}
		else if (effect > 0.25)
		{
			hatchVal = hatch.g;
		}
		else
		{
			hatchVal = hatch.b;
		}
		fColor = vec4(hatchVal * result, 1);
	}
	else
	{
		fColor = vec4(result, 1.0);
	}
}