#version 440 core
#include "lighting"
#include "camera"
#include "forward_buffers"

layout(location = 0) in vec3 fPos;
layout(location = 1) in vec2 fTex;
layout(location = 2) in mat3 TBNmat;

layout(location = 0) out vec4 fragmentColor;

layout(location = 1) uniform vec3 shapeColor;
layout(location = 2) uniform int checkUVs;

layout(location = 0) uniform sampler2D color;


uniform int TileSize; 
uniform vec2 ScreenSize;
uniform uvec2 tileDimension;

void main()
{
	// TODO: Texture reads for maps and stuff
	vec3 norm = vec3(0, 0, 1);
	vec3 mod = (gl_FrontFacing) ? vec3(1.f) : vec3(-1.f);
	
	vec3 viewDirection = normalize(View[3].xyz - fPos);
		
	// This is a hack, but for some reason gl_FrontFacing won't work otherwise. Need to work on this
	vec3 normal = (vec4(TBNmat * norm, 0)).xyz;
	
	
	vec2 index = floor((gl_FragCoord.xy) / TileSize);
	
	uint gridIndex = uint(index.x + index.y * tileDimension.x);
	
	uvec2 lightData = grid[gridIndex];
	
	float ambient = 0.15;
	
	vec3 lightOut = vec3(ambient);
	
	for (int i = 0; i < lightData.y; i++)
	{
		uint index = indicies[i + lightData.x];
		LightInfoBig current = lightsOriginal[index];
		if (length(current.position.xyz - fPos) > current.position.w)
			continue;
		lightOut += PointLightStruct(current, normal, fPos, viewDirection);
	}
	
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
	//fragmentColor = vec4(float(lightData.y) / 60, 0, 0, 1);
}
