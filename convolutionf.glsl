#version 440 core

in vec2 textureCoords;
out vec4 fColor;

layout(location = 0) uniform sampler2D screen;
layout(location = 1) uniform sampler2D convolution;

void main()
{
	const ivec2 kernel = textureSize(convolution, 0);
	for (int i = 0; i < kernel.x; i++)
	{
		float u = i - floor(kernel.x / 2);
		for (int j = 0; j < kernel.y; j++)
		{
			float v = j - floor(kernel.y / 2);
			vec2 delta = vec2(u, v);
			fColor += textureOffset(screen, textureCoords, delta) * texelFetch(convolution, ivec2(i, j));	
		}
	}
	fColor = texture(screen, textureCoords);
}