#version 440 core

in vec2 textureCoords;
layout(location = 0) out vec4 fColor;


layout(location = 0) uniform sampler2D screen;
layout(location = 1) uniform sampler2D edges;
layout(location = 2) uniform sampler2D depths;
layout(location = 3) uniform usampler2D stencil;

uniform int depth;
uniform int flag;



void main()
{
	vec4 sampled = texture(screen, textureCoords);
	
	uint samp = texture(stencil, textureCoords).r;
	float sten = float(texture(stencil, textureCoords).r);
	vec3 fool;
	
	if (samp == 2)
	{
		fool = vec3(1, 0, 0);
	}
	else if (samp == 1)
	{
		fool = vec3(0, 1, 0);
	}
	else if (samp == 0)
	{
		fool = vec3(0.5, 0.5, 0.5);
	}
	else
	{
		fool = vec3(1, 1, 1);
	}
	
	fColor = (flag > 0) ? vec4(fool * sampled.xyz, 1) : sampled;
	fColor.w = 1;
}