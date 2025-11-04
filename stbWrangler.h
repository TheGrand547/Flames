#pragma once
#ifndef STB_WRANGLER_H
#define STB_WRANGLER_H

#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG

#include <stdio.h>
#include <string.h>
#pragma warning(push)
#pragma warning(disable : 4100)
#include "stb_image.h"
#include "stb_perlin.h"
#include "stb_rect_pack.h"
#include "stb_truetype.h"
#pragma warning(pop)
#endif