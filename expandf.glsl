#version 440 core

in vec2 textureCoords;
out vec4 fColor;


layout(location = 0) uniform sampler2D screen;
layout(location = 1) uniform sampler2D edges;

uniform int depth;

bool dark = false;

void test(ivec2 offset)
{
	if (!dark && textureOffset(edges, textureCoords, offset).r < 0.25)
	{
		dark = true;
	}
}

void main()
{
	vec4 sampled = texture(screen, textureCoords);
	test(ivec2(0, 0));
	for (int i = 1; i <= depth && !dark; i++)
	{
		test(ivec2( i,  0));
		test(ivec2(-i,  0));
		test(ivec2( 0,  i));
		test(ivec2( 0, -i));
		test(ivec2(-i, -i));
		test(ivec2( i, -i));
		test(ivec2(-i,  i));
		test(ivec2( i,  i));
	}
	if (dark)
	{
		fColor = vec4(0.15, 0.15, 0.15, 1);
	}
	else
	{
		fColor = vec4(1, 1, 1, 1);
	}
	fColor = sampled * fColor;
	//fColor = texture(edges, textureCoords);
	//fColor.w = 1;
}