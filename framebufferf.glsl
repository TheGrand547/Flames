#version 440 core

in vec2 textureCoords;
out vec4 fColor;

layout(location = 0) uniform sampler2D screen;
layout(location = 1) uniform sampler2D normal;

void main()
{
	const float offset = 1.0 / textureSize(screen, 0).x;
	vec2 offsets[9] = vec2[](
		vec2(-offset,  offset), // top-left
		vec2( 0.0f,	   offset), // top-center
		vec2( offset,  offset), // top-right
		vec2(-offset,  0.0f),	// center-left
		vec2( 0.0f,	   0.0f),	// center-center
		vec2( offset,  0.0f),	// center-right
		vec2(-offset, -offset), // bottom-left
		vec2( 0.0f,	  -offset), // bottom-center
		vec2( offset, -offset)	// bottom-right
	);
	
	
	float kernel[9] = float[](
		-1, -1, -1,
		-1,	 8, -1,
		-1, -1, -1
	);

	fColor = vec4(0, 0, 0, 1);
	for(int i = 0; i < 9; i++)
	{
		fColor += vec4(vec3(texture(normal, textureCoords + offsets[i])) * kernel[i], 0);
	}
	float large = max(abs(fColor.x), max(abs(fColor.y), abs(fColor.z)));
	
	fColor = 1 - vec4(step(0.25, large));
	fColor.w = 1;
	
	//	fColor = 1 - vec4(1, 1, 1, 0) * max(fColor.x, max(fColor.y, fColor.z));
	//fColor = fColor * texture(screen, textureCoords);
	//fColor = fColor + texture(screen, textureCoords);
	//fColor = 1 - vec4(1, 1, 1, 0) * step(0.125, max(fColor.x, max(fColor.y, fColor.z)));
	// Leads to the cool "dark world" effect
	//fColor = 1 - vec4(1, 1, 1, 0) * step(0.125, max(fColor.x, max(fColor.y, fColor.z)));
	//fColor = fColor + texture(screen, textureCoords);
}