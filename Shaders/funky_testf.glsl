#version 440 core

layout(location = 0) in vec2 textureCoords;
layout(location = 0) out vec4 fColor;

layout(location = 0) uniform usampler2D stencil;
layout(location = 1) uniform sampler2D rainbow;

vec3 colors[] = {
	vec3(1, 0, 0), vec3(0, 1, 0),
	vec3(0, 0, 1), vec3(0, 1, 1)
};


void main()
{
	uint sampled = texture(stencil, textureCoords).r;
	if (sampled == 0)
		discard;
	fColor.rgb = texture(rainbow, vec2(float(sampled) / 255, 0)).rgb;
	fColor.a = 0.5f;
}