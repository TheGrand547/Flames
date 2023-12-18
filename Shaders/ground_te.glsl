#version 440 core

vec3 _interpolate3(in vec3 a, in vec3 b, in vec3 c, in vec3 d)
{
	return mix(mix(a, b, gl_TessCoord.x), mix(c, d, gl_TessCoord.x), gl_TessCoord.y);
}
#define interpolate3(array) _interpolate3(array[0], array[1], array[2], array[3])

layout (quads, equal_spacing, ccw) in;

layout(location = 0) in vec3 tePos[];
layout(location = 1) in vec3 teNorm[];
layout(location = 2) in vec2 teTex[];

layout(location = 0) out vec3 fPos;
layout(location = 1) out vec3 fNorm;
layout(location = 2) out vec2 fTex;

layout(std140) uniform Camera
{
	mat4 View;
	mat4 Projection;
};

uniform sampler2D heightMap;


void main()
{
	vec2 uv = gl_TessCoord.xy;
	fPos = interpolate3(tePos);
	fNorm = interpolate3(teNorm);

	fPos.y += texture(heightMap, uv).r - 0.5;

	gl_Position = Projection * View * vec4(fPos, 1);
}