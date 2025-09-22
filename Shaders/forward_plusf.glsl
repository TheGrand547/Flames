#version 440 core

#include "camera"
#include "forward_buffers"

layout(location = 0) in vec3 fPos;
layout(location = 1) in vec2 fTex;
layout(location = 2) in mat3 TBNmat;

layout(location = 0) out vec4 fragmentColor;

layout(location = 1) uniform vec3 shapeColor;
layout(location = 2) uniform int checkUVs;

void main()
{
	// TODO: Texture reads for maps and stuff
	vec3 norm = vec3(0, 0, 1);
	vec3 mod = (gl_FrontFacing) ? vec3(1.f) : vec3(-1.f);
	
	vec3 viewPos = (View * vec4(fPos, 1).xyz);
	
	
	// This is a hack, but for some reason gl_FrontFacing won't work otherwise. Need to work on this
	fragmentNormal = vec4(TBNmat * norm * mod, 1);
	
	vec4 sampled = texture(textureColor, fTex);
	
	if (checkUVs == 1)
	{
		vec2 big = floor(fTex * 10);
		bool flip = (int(big.x) % 2 == 0) ^^ (int(big.y) % 2 == 0);
		float mult = (flip) ? 1.f : 0.5f;
		sampled.xyz *= mult;
	}
	fragmentColor = vec4(shapeColor * sampled.xyz, 1);
}
