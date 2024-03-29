#version 440 core

in vec3 pos;
in vec2 tex;

out vec2 texCoord;

uniform mat4 mvp;

void main()
{
	gl_Position = mvp * vec4(pos, 1.0);
	texCoord = tex;
}