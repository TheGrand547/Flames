#pragma once
#ifndef TEXTURE_UTIL_H
#define TEXTURE_UTIL_H
#include "Texture.h"
#include "Buffer.h"
#include "ScreenRect.h"
#include "Texture2D.h"

namespace Texture
{
	
}
std::array<ScreenRect, 9> NineSliceGenerate(glm::ivec2 topLeft, glm::ivec2& size);

Texture2D HeightToNormal();

#endif // TEXTURE_UTIL_H

