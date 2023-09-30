#version 440 core

out vec4 vColor;

vec3 positions[] = {
	vec3(0.f, 0.f, 0.f), vec3(1.f, 0.f, 0.f),
	vec3(0.f, 0.f, 0.f), vec3(0.f, 1.f, 0.f),
	vec3(0.f, 0.f, 0.f), vec3(0.f, 0.f, 1.f),
};

vec4 colors[] = {
	vec4(1.f, 0.f, 0.f, 1.f), 
	vec4(0.f, 1.f, 0.f, 1.f),
	vec4(0.f, 0.f, 1.f, 1.f)
};

layout(std140) uniform Camera
{
	mat4 View;
	mat4 Projection;
};

void main()
{
	gl_Position = View * vec4(positions[gl_VertexID % 6] * 0.1f, 0);
	gl_Position -= vec4(0.9);
	gl_Position.w = 1;
	vColor = colors[int(gl_VertexID / 2) % 3];
}