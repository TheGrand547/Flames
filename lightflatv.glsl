#version 440 core

in vec3 vPos;
in vec3 vNorm;

out vec4 fNorm;
out vec3 fPos;
out vec3 color;

uniform mat4 modelMat;
uniform mat4 normMat;
uniform mat4 viewProjMat; 

uniform vec3 lightColor;
uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 shapeColor;

void main()
{
	fNorm = normMat * vec4(vNorm, 0);
	
	fPos = vec3(modelMat * vec4(vPos, 1.0));
	gl_Position = viewProjMat * modelMat * vec4(vPos, 1.0);

		/*
	float ambient = 0.2f; // TODO: material setting
	vec3 ambientColor = lightColor * ambient;
	
	vec3 norm = fNorm.xyz;
	vec3 lightDir = normalize(lightPos - fPos);
	
	float diffuse = max(dot(norm, lightDir), 0.0);
	vec3 diffuseColor = diffuse * lightColor;
	
	vec3 viewDirection = normalize(viewPos - fPos);
	vec3 reflected = reflect(-lightDir, norm);

	float specular = pow(max(dot(viewDirection, reflected), 0.0), 128); // TODO: Specular setting
	vec3 specularOut = lightColor * specular; // TODO: I don't remember
*/
	color = shapeColor;// * (ambientColor + diffuseColor + specularOut);
}