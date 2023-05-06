#version 440 core

in vec2 textureCoords;
out vec4 fColor;

uniform sampler2D screen;

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
		fColor += vec4(vec3(texture(screen, textureCoords + offsets[i])) * kernel[i], 0);
	}
	fColor = 1 - vec4(1, 1, 1, 0) * step(0.25, max(fColor.x, max(fColor.y, fColor.z)));
	//fColor = texture(screen, textureCoords);
}