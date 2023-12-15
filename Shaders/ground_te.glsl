#version 440 core

vec3 interpolate(in vec3 a, in vec3 b, in vec3 c, in vec3 d, in vec2 coords);

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
	fPos =  interpolate(tePos[0], tePos[1], tePos[2], tePos[3], uv);
	fNorm =  interpolate(teNorm[0], teNorm[1], teNorm[2], teNorm[3], uv);
	//fTex =  interpolate(teTex[0], teTex[1], teTex[2], teTex[3], uv);

	fPos.y = texture(heightMap, uv).r;

	gl_Position = Projection * View * vec4(fPos, 1);
}

vec3 interpolate(in vec3 a, in vec3 b, in vec3 c, in vec3 d, in vec2 coords)
{
	vec3 foo =  mix(mix(a, b, coords.x), mix(c, d, coords.x), coords.y);
	return foo;
}