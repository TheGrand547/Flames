#version 440 core

// TODO: Put these in the funny glsl header
layout(location = 0) in vec3 vPos;
layout(location = 1) in vec3 vNorm;
layout(location = 2) in vec2 vTex;

out vec4 fNorm;
out vec3 fPos;
out vec3 color;

uniform mat4 modelMat;
uniform mat4 normalMat;

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
	fNorm = normalMat * vec4(vNorm, 0);
	
	// TODO: Recheck this later, very good
	// Changing this number is very funny, but keep it small
	vec3 zoop = normalize(vNorm) * int(int(gl_VertexID) % 5 < 1) * 0.25;
	if (int(gl_VertexID) % 7 == 12)
		zoop *= 0.5;
	zoop = vec3(0,0,0);
	fPos = vec3(modelMat * vec4(vPos, 1.0));
	
	gl_Position = Projection * View * (modelMat * vec4(vPos + zoop, 1.0));

	color = shapeColor;// * (ambientColor + diffuseColor + specularOut);
}