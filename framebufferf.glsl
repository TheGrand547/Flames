#version 440 core

in vec2 textureCoords;
out vec4 fColor;

layout(location = 0) uniform sampler2D normal;
layout(location = 1) uniform sampler2D depth;

layout(location = 2) uniform float zNear;
layout(location = 3) uniform float zFar;
uniform int zoop;

void main()
{	
	float kernel[9];
	if (zoop != 1) 
	{
		kernel = float[](
			-1, -1, -1,
			-1,	 8, -1,
			-1, -1, -1
		);
	}
	else
	{
		kernel = float[](
			-1.5, -1, -1.5,
			-1,	 10, -1,
			-1.5, -1, -1.5
		);
	}

	fColor = vec4(0, 0, 0, 1);
	float depthDelta = 0.f;
	
	for(int i = 0; i < 9; i++)
	{
		ivec2 offset = ivec2((i % 3) - 1, floor(i / 3) - 1);
		float kernelValue = kernel[i];
		float depthSample = textureOffset(depth, textureCoords, offset).r;
		fColor += textureOffset(normal, textureCoords, offset) * kernelValue * depthSample;
		depthSample = 2 * depthSample - 1;
		depthSample = 2.0 * zNear * zFar / (zFar + zNear - depthSample * (zFar - zNear));
		//depthDelta += depthSample * kernelValue;
	}
	float large = max(abs(fColor.x), max(abs(fColor.y), abs(fColor.z)));
	
	// Calculating "true" difference in depth, world coordinates -- horrendously ugly
	if (abs(depthDelta) > 1)
		large = 1.f;
	
	fColor = 1 - vec4(step(0.25, large));
	
	
	//fColor = 1 - vec4(step(0.25, abs(depthDelta)));
	//fColor = 1 - vec4(large);
	fColor.w = 1;
	
	
	
	//fColor = abs(fColor);
	//fColor = abs(texture(normal, textureCoords);
	//fColor = fColor * texture(screen, textureCoords);
	
	
	//fColor = 1 - vec4(1, 1, 1, 0) * step(0.125, max(fColor.x, max(fColor.y, fColor.z)));
	// Leads to the cool "dark world" effect
	//fColor = 1 - vec4(1, 1, 1, 0) * step(0.125, max(fColor.x, max(fColor.y, fColor.z)));
	//fColor = fColor + texture(screen, textureCoords);
}