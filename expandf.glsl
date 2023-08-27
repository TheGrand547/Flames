#version 440 core

in vec2 textureCoords;
out vec4 fColor;


layout(location = 0) uniform sampler2D screen;
layout(location = 1) uniform sampler2D edges;
layout(location = 2) uniform sampler2D depths;

uniform int depth;

int dark = 0;
const int required = 3;

void test(ivec2 offset)
{
	if (dark < required && textureOffset(edges, textureCoords, offset).r < 0.25)
	{
		dark += 1;
	}
}

void main()
{
	vec4 sampled = texture(screen, textureCoords);
	test(ivec2(0, 0));
	/*
	for (int i = 1; i <= depth && dark <= required; i++)
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
	*/
	for (int x = 0; x <= depth && dark <= required; x++)
	{
		int m_x = x - depth / 2;
		for (int y = 0; y <= depth && dark <= required; y++)
		{
			int m_y = y - depth / 2;
			test(ivec2(m_x, m_y));
		}
	}
	if (dark >= required)
	{
		//fColor = vec4(0.15, 0.15, 0.15, 1);
		fColor = vec4(1, 1, 1, 1);
	}
	else
	{
		//fColor = vec4(1, 1, 1, 1);
		//fColor = sampled;
		fColor = sampled + (float(dark) / required) * vec4(1, 1, 1, 1);
	}
	//fColor = sampled * fColor;
	fColor.w = 1;
}