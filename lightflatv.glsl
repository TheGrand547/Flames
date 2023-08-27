#version 440 core

in vec3 vPos;
in vec3 vNorm;

out vec4 fNorm;
out vec3 fPos;
out vec3 color;

uniform mat4 modelMat;
uniform mat4 normMat;

layout(std140) uniform Camera
{
	mat4 View;
	mat4 Projection;
};

uniform vec3 lightColor;
uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 shapeColor;

void main()
{
	fNorm = normalize(normMat * vec4(vNorm, 0));
	
	fPos = vec3(modelMat * vec4(vPos, 1.0));
	
	// TODO: Recheck this later, very good
	// Changing this number is very funny, but keep it small
	vec3 zoop = vNorm * int(int(gl_VertexID) % 5 == 0) * 0.5 * 0;
	if (int(gl_VertexID) % 7 == 12)
		zoop *= 0.5;
	
	gl_Position = Projection * View * (modelMat * vec4(vPos + vec3(zoop), 1.0));

	color = shapeColor;// * (ambientColor + diffuseColor + specularOut);
}