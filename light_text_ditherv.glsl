#version 440 core

in vec3 pos;
in vec2 texturePos;

out vec3 normal;
out vec3 fragPos;
out vec2 tex;

uniform mat4 model;
uniform mat4 vp; 
// TODO: Normal matrix

void main()
{
	// TODO: Add normal as a vertex attribute and have the rotation/translation matrix as a separate parameter?
	//Normal = mat3(transpose(inverse(model))) * aNormal;  
	//normal = normalize(vec3(model * vec4(0, 1, 0, 0)));
	normal = normalize(mat3(transpose(inverse(model))) * vec3(0, 1, 0));
	fragPos = vec3(model * vec4(pos, 1.0));
	gl_Position = vp * model * vec4(pos, 1.0);
	tex = texturePos;
}