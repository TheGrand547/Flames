#version 440 core

in vec2 textureCoords;
layout(location = 0) out vec4 fColor;


layout(location = 0) uniform sampler2D screen;
layout(location = 1) uniform sampler2D edges;
layout(location = 2) uniform sampler2D depths;
layout(location = 3) uniform usampler2D stencil;

uniform int depth;

int dark = 0;
const int required = 4;

void test(ivec2 offset)
{
	if (dark < required && textureOffset(edges, textureCoords, offset).r < 0.25)
	{
		dark += 1;
	}
}


float guassian[] = float[] (
-1, -1, -1, -1, -1,
-1, -1, -1, -1, -1,
-1, -1, 24, -1, -1,
-1, -1, -1, -1, -1,
-1, -1, -1, -1, -1
);
/* float[](
			-1.5, -1, -1.5,
			-1,	 10, -1,
			-1.5, -1, -1.5
		);*/
/*
float[] 
(
	1,  4,  6,  4, 1,
	4, 16, 24, 16, 4,
	6, 24, 36, 24, 6,
	4, 16, 24, 16, 4,
	1,  4,  6,  4, 1
);
*/
void main()
{
	// TODO: GUASSIAN INSTEAD BECAUSE THAT'S COOLER
	vec4 sampled = texture(screen, textureCoords);
	test(ivec2(0, 0));
	
	float blurred = 0.f;
	vec4 mid = vec4(0, 0, 0, 0);
	
	int sizer = 5;
	for (int x = 0; x < sizer; x++)
	{
		int m_x = x - (sizer / 2);
		for (int y = 0; y < sizer; y++)
		{
			int m_y = y - (sizer / 2);
			
			mid += textureOffset(edges, textureCoords, ivec2(m_x, m_y)) * guassian[x * sizer + y];
			// guassian[x * 5 + y]
			//blurred += (1 - textureOffset(edges, textureCoords, ivec2(m_x, m_y)).r) / 25.f;
			//blurred += atan(1 - textureOffset(edges, textureCoords, ivec2(m_x, m_y)).r) * guassian[x * 5 + y] / 256.f;
		}
	}
	mid = abs(mid);
	blurred = max(mid.x, max(mid.y, mid.z));
	fColor = vec4(length(mid));
	//fColor = mix(sampled, fColor, pow(blurred, 0.9f));
	fColor = sampled;
	//fColor = vec4(step(0.2, length(mid)));
	//fColor = abs(mid);
	
	//fColor = exp(-fColor);
	//blurred = step(.2, blurred) * blurred;
	//fColor = sampled + vec4(pow(blurred, 1.5f));
	/*
	if (dark >= required)
	{
		//fColor = vec4(0.15, 0.15, 0.15, 1);
		fColor = vec4(1, 1, 1, 1);
	}
	else
	{
		//fColor = vec4(1, 1, 1, 1);
		fColor = sampled;
		//fColor = sampled + (float(dark) / required) * vec4(1, 1, 1, 1);
	}*/
	//fColor = sampled * fColor;
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
		fool = vec3(0, 0, 0.5);
	}
	else
	{
		fool = vec3(1, 1, 1);
	}
	fColor.xyz += fool;
	//fColor.xyz = mix(vec3(0, 0, 0), fColor.xyz, sten);
	//fColor.xyz = vec3(1, 0.25, 0.5);
	
	fColor.w = 1;
}