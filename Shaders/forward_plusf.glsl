#version 440 core
#include "lighting"
#include "camera"
#include "forward_buffers"
#include "forward_plus"

layout(location = 0) in vec3 fPos;
layout(location = 1) in vec2 fTex;
layout(location = 2) in mat3 TBNmat;

layout(location = 0) out vec4 fragmentColor;

layout(location = 1) uniform vec3 shapeColor;
layout(location = 2) uniform int checkUVs;

layout(location = 0) uniform sampler2D color;

void main()
{
	// TODO: Texture reads for maps and stuff
	vec3 norm = vec3(0, 0, 1);
	vec3 mod = (gl_FrontFacing) ? vec3(1.f) : vec3(-1.f);
	
	vec3 viewDirection = normalize(View[3].xyz - fPos);
		
	// This is a hack, but for some reason gl_FrontFacing won't work otherwise. Need to work on this
	vec3 normal = (vec4(TBNmat * norm, 0)).xyz;
	
	vec3 lightOut = ForwardPlusLighting(fPos, normal, viewDirection);
	
	// For textured stuff
	//vec4 sampled = texture(textureColor, fTex);
	vec4 sampled = vec4(1);
	if (checkUVs == 1)
	{
		vec2 big = floor(fTex * 10);
		bool flip = (int(big.x) % 2 == 0) ^^ (int(big.y) % 2 == 0);
		float mult = (flip) ? 1.f : 0.5f;
		sampled.xyz *= mult;
	}
	fragmentColor = vec4(shapeColor * sampled.xyz * lightOut, 1);
	//fragmentColor = vec4(LightTesting(), 1.f);
	//fragmentColor = vec4(index / tileDimension, 0, 1);
}
