#version 440 core

in vec3 pos;
in mat4 vert;
out vec4 colorOut;

uniform mat4 vp;
uniform vec3 color;

void main()
{
	gl_Position = vert * vp * vec4(pos, 1.0);
	colorOut = vec4(color, 1.0);
}