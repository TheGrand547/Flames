#version 440 core

layout (triangles, equal_spacing, ccw) in;

layout(location = 0) in vec3 fPos2[];
layout(location = 1) in vec3 fNorm2[];
layout(location = 2) in vec2 fTex2[];

layout(location = 0) out vec3 fPosF;
layout(location = 1) out vec3 fNormF;
layout(location = 2) out vec2 fTexF;

layout(std140) uniform Camera
{
	mat4 View;
	mat4 Projection;
};

void main()
{
	vec2 uv = gl_TessCoord.xy;
	fPosF = mix(mix(fPos2[0], fPos2[1], uv.x), fPos2[2], uv.y);
	fNormF = mix(mix(fNorm2[0], fNorm2[1], uv.x), fNorm2[2], uv.y);
	fTexF = mix(mix(fTex2[0], fTex2[1], uv.x), fTex2[2], uv.y);

	vec3 accum = vec3(0.0f);
	accum += gl_TessCoord[0] * fPos2[0];
	accum += gl_TessCoord[1] * fPos2[1];
	accum += gl_TessCoord[2] * fPos2[2];
	gl_Position = Projection * View * vec4(accum, 1); //mix(gl_in[2].gl_Position, mix(gl_in[0].gl_Position, gl_in[0].gl_Position, uv.x), uv.y);
}