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
	float kernel[25];
	// 5x5 Sharpen Kernel
	kernel = float[](
			-1, -1, -1, -1, -1,
			-1, -1, -1, -1, -1,
			-1, -1, 24, -1, -1,
			-1, -1, -1, -1, -1,
			-1, -1, -1, -1, -1
			);
	// 5x5 Unsharp Masking
	kernel = float[]
	(
		-1,  -4,  -6,  -4, -1,
		-4, -16, -24, -16, -4,
		-6, -24, 476-256, -24, -6,
		-4, -16, -24, -16, -4,
		-1,  -4,  -6,  -4, -1
	);
	// Approximated 5x5 Guassian Kernel
	kernel = float[] 
	(
		1,  4,  6,  4, 1,
		4, 16, 24, 16, 4,
		6, 24, 36, 24, 6,
		4, 16, 24, 16, 4,
		1,  4,  6,  4, 1
	);
	
			

	fColor = vec4(0, 0, 0, 1);
	float depthDelta = 0.f;
	float dev = 9;
	float dev2 = dev * dev;
	float constant = inversesqrt(2 * acos(-1) * dev2);

	for(int i = 0; i < 25; i++)
	{
		ivec2 offset = ivec2((i % 5) - (5/2), floor(i / 5) - (5/2));
		float kernelValue = kernel[i];
		float depthSample = textureOffset(depth, textureCoords, offset).r;
		//fColor += textureOffset(normal, textureCoords, offset) * kernelValue * depthSample;
		fColor += textureOffset(normal, textureCoords, offset) * kernelValue / 256.f;
		//fColor += textureOffset(normal, textureCoords, offset) * constant * exp(-length(offset) / dev2);
		//fColor += textureOffset(normal, textureCoords, offset) / 25.f;
		
		
		depthSample = 2 * depthSample - 1;
		depthSample = 2.0 * zNear * zFar / (zFar + zNear - depthSample * (zFar - zNear));
		//depthDelta += depthSample * kernelValue;
	}
	float large = max(abs(fColor.x), max(abs(fColor.y), abs(fColor.z)));
	
	// Calculating "true" difference in depth, world coordinates -- horrendously ugly
	//if (abs(depthDelta) > 1)
		//large = 1.f;
	
	//fColor = 1 - vec4(step(0.25, large));
	//fColor = 1 - vec4(large);
	
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