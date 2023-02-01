#version 440 core

in vec3 pos;
in vec3 color;
out vec4 colorOut;

uniform mat4 mvp;
void main()
{
	gl_Position = mvp * vec4(pos, 1.0);
	colorOut = vec4(color, 1.0);
}