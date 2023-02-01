#version 440 core

in vec3 pos;
out vec3 colorOut;
out vec3 normal;
out vec3 fragPos;

uniform mat4 model;
uniform mat4 vp;
uniform vec3 color;

void main()
{
	normal = normalize(vec3(model * vec4(0, 1, 0, 0)));
	fragPos = vec3(model * vec4(pos, 1.0));
	gl_Position = vp * model * vec4(pos, 1.0);
	colorOut = color;
}