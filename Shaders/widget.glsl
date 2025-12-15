#version 440 core
#include "camera"

#ifdef VERTEX

layout(location = 0) out vec4 vColor;

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

void main()
{
	const uint index = gl_VertexID % 6;
	gl_Position = View * vec4(positions[index] * 0.1f, 0);
	gl_Position.xy -= vec2(0.9);
	
	// Technically shouldn't be necessary(depth testing off), but why not be sure
	gl_Position.zw = vec2(1, 1);
	vColor = colors[index >> 1];
}

#endif // VERTEX

#ifdef FRAGMENT

layout(location = 0) in vec4 vColor;

layout(location = 0) out vec4 fColor;

void main()
{	
	fColor = vColor;
}

#endif // FRAGMENT
