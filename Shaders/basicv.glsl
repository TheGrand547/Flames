#version 440 core

layout(location = 0) in vec3 vPos;

uniform mat4 Model;

layout(std140, binding = 0) uniform Camera
{
	mat4 View;
	mat4 Projection;
};

void main()
{
	gl_Position = Projection * View * Model * vec4(vPos, 1.0);
}