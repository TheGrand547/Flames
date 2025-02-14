#version 440 core

layout(location = 0) in vec3 vPos;

layout(location = 0) out vec4 fColor;

layout(std140, binding = 0) uniform Camera
{
	mat4 View;
	mat4 Projection;
};

uniform vec3 Color;

const int duration = 256;

void main()
{
	gl_Position = Projection * View * vec4(vPos, 1.0);
	fColor.xyz = Color;
	fColor.w = float(gl_VertexID) / duration;
	fColor.w *= fColor.w;
}