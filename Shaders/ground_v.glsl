#version 440 core

layout(location = 0) in vec3 vPos;
layout(location = 1) in vec2 vTex;


layout(location = 0) out vec3 tePos;
layout(location = 1) out vec3 teNorm;
layout(location = 2) out vec2 teTex;

uniform mat4 Model;
layout(std140) uniform Camera
{
	mat4 View;
	mat4 Projection;
};

void main()
{
	gl_Position = Model * vec4(vPos, 1);
	tePos = gl_Position.xyz;
	teNorm = (View * gl_Position).xyz;
	teTex = vTex;
}