#version 440 core
#include "lighting"
#include "forward_buffers"
layout(location = 0) in vec2 textureCoords;
layout(location = 0) out vec4 fColor;

uniform int TileSize; 
uniform vec2 ScreenSize;
uniform uvec2 tileDimension;
uniform int maxLight;

layout(origin_upper_left) in vec4 gl_FragCoord;


void main()
{
	//vec2 index = floor((gl_FragCoord.xy) / TileSize);
	vec2 index = floor((gl_FragCoord.xy) / TileSize);
	
	uint gridIndex = uint(index.x + index.y * tileDimension.x);
	uvec2 lightData = grid[gridIndex];
	if (lightData.y > maxLight)
	{
		fColor = vec4(0, 0, 1, 1);
	}
	else
	{
		fColor = vec4(float(lightData.y) / maxLight, 0, 0, 1);
	}
	//fColor = vec4(index / tileDimension, 0, 1);
}
