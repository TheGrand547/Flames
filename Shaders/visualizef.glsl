#version 440 core
#include "lighting"
#include "forward_buffers"
layout(location = 0) in vec2 textureCoords;
layout(location = 0) out vec4 fColor;

uniform int maxLight;

layout(origin_upper_left) in vec4 gl_FragCoord;


void main()
{
	vec2 index = floor((gl_FragCoord.xy) / TileSize);
	
	uint gridIndex = uint(index.x + index.y * tileDimension.x);
	
	uint numLightsThisTile = tileMasks[gridIndex * MasksPerTile];
	
	if (numLightsThisTile > maxLight)
	{
		fColor = vec4(0, 0, 1, 1);
	}
	else
	{
		fColor = vec4(float(numLightsThisTile) / maxLight, 0, 0, 1);
	}
}
