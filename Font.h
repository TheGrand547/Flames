#pragma once
#ifndef FONT_H
#define FONT_H
#include <array>
#include "stbWrangler.h"
#include "Texture2D.h"

class Font
{
protected:
	std::array<stbtt_packedchar, 96> characters;
	float pixelHeight, scalingFactor;

	bool sharedAtlas = false;
	Texture2D texture; // TODO: Maybe shared texture class or smth?, maybe just a shared pointer who knows
public:


	static void SetFontDirectory(const std::string& directory);
	static bool LoadFont(Font& font, const std::string& filename, float fontSize, unsigned int sampleX = 1, unsigned int sampleY = 1, int padding = 1);
};

#endif // FONT_H