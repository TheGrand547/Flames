#version 440 core

in vec2 textureCoords;
out vec4 fColor;

layout(location = 0) uniform sampler2D normal;

void main()
{	
	float kernel[9] = float[](
		-1, -1, -1,
		-1,	 8, -1,
		-1, -1, -1
	);

	fColor = vec4(0, 0, 0, 1);
	for(int i = 0; i < 9; i++)
	{
		fColor += textureOffset(normal, textureCoords, ivec2((i % 3) - 1, floor(i / 3) - 1)) * kernel[i];
	}
	float large = max(abs(fColor.x), max(abs(fColor.y), abs(fColor.z)));
	
	//fColor = 1 - vec4(step(0.5, large));
	fColor = 1 - vec4(large);
	fColor.w = 1;
	//fColor = abs(fColor);
	//fColor = abs(texture(normal, textureCoords);
	//fColor = fColor * texture(screen, textureCoords);
	
	
	//fColor = 1 - vec4(1, 1, 1, 0) * step(0.125, max(fColor.x, max(fColor.y, fColor.z)));
	// Leads to the cool "dark world" effect
	//fColor = 1 - vec4(1, 1, 1, 0) * step(0.125, max(fColor.x, max(fColor.y, fColor.z)));
	//fColor = fColor + texture(screen, textureCoords);
}