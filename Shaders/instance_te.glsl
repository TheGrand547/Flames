#version 440 core

layout (quads, fractional_odd_spacing, ccw) in;

layout(location = 0) in vec3 fPos2[];
layout(location = 1) in vec3 fNorm2[];
layout(location = 2) in vec2 fTex2[];
layout(location = 3) in mat4 Model3[];


layout(location = 0) out vec3 fPos3;
layout(location = 1) out vec3 fNorm3;
layout(location = 2) out vec2 fTex3;



layout(std140) uniform Camera
{
	mat4 View;
	mat4 Projection;
};

void main()
{
	vec2 uv = gl_TessCoord.xy;
	
	fPos3 = mix(mix(fPos2[0], fPos2[1], uv.x), mix(fPos2[2], fPos2[3], uv.x), uv.y);
	fNorm3 = mix(mix(fNorm2[0], fNorm2[1], uv.x), mix(fNorm2[2], fNorm2[3], uv.x), uv.y);
	fTex3 = mix(mix(fTex2[0], fTex2[1], uv.x), mix(fTex2[2], fTex2[3], uv.x), uv.y);
	
	gl_Position = Projection * View * Model3[0] * gl_Position;
}