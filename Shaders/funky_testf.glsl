#version 440 core

layout(location = 0) in vec2 textureCoords;
layout(location = 0) out vec4 fColor;

layout(location = 0) uniform usampler2D stencil;

vec3 colors[] = {
	vec3(1, 0, 0), vec3(0, 1, 0),
	vec3(0, 0, 1), vec3(0, 1, 1)
};


void main()
{
	uint sampled = texture(stencil, textureCoords).r;
	uint mask = 0x3;
	vec3 total = vec3(0);
	float largest = 0;
	int hits = 0;
	for (int i = 0; i < 4; i++)
	{
		uint current = (sampled >> (2 * i)) & mask;
		if (current != 0) 
			hits = hits + 1;
		float ratio = current / 3.f;
		total += ratio * colors[i];
		largest = max(largest, ratio);
	}
	fColor = vec4(total / hits, largest);
}