#include "TextureUtil.h"
#include "glmHelp.h"
/*
20x20 tiles(for the moment) 


*/

// Takes the desired size
std::array<ScreenRect, 9> NineSliceGenerate(glm::ivec2 topLeft, glm::ivec2& size)
{
	const glm::ivec2 tileSize(20, 20);
	size = glm::max(size, 2 * tileSize);
	const glm::ivec2 clipped = size - 2 * tileSize;
	const glm::ivec2 edge = size - tileSize;
	// Assume size is bigger than the minimum 40x40
	std::array<ScreenRect, 9> rects =
	{
		{
			{glm::vec4(0, 0, tileSize)}, 
			{glm::vec4(tileSize.x, 0, clipped.x, tileSize.y)},
			{glm::vec4(edge.x, 0, tileSize)},
			{glm::vec4(0, tileSize.y, tileSize.x, clipped.y)},
			{glm::vec4(tileSize, clipped)},
			{glm::vec4(edge.x, tileSize.y, tileSize.x, clipped.y)},
			{glm::vec4(0, edge.y, tileSize)},
			{glm::vec4(tileSize.x, edge.y, clipped.x, tileSize.y)},
			{glm::vec4(edge, tileSize)}
		}
	};
	for (auto& ref : rects)
	{
		ref += glm::vec4(topLeft, 0, 0);
	}
	return rects;
}