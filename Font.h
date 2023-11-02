#pragma once
#ifndef FONT_H
#define FONT_H
#include <array>
#include "Buffer.h"
#include "stbWrangler.h"
#include "Texture2D.h"

namespace Font
{
	void SetFontDirectory(const std::string& directory);
}

class ASCIIFont
{
protected:
	std::array<stbtt_packedchar, '~' - ' '> characters;
	float pixelHeight, scalingFactor;
	int ascender, descender, lineGap;
	int lineSkip;

	Texture2D texture; // TODO: Maybe shared texture class or smth?, maybe just a shared pointer who knows
public:
	constexpr ASCIIFont();

	inline const Texture2D& GetTexture() const;

	// Provide a buffer filled with the position and texture coordinates that with this font's texture can be draw to whatever target
	void Render(Buffer<ArrayBuffer>& buffer, float x, float y, const std::string& message);
	void Render(Buffer<ArrayBuffer>& buffer, const glm::vec2& coords, const std::string& message);

	// Renders directly to the texture
	void Render(Texture2D& texture, float x, float y, const std::string& message);
	void Render(Texture2D& texture, const glm::vec2& coords, const std::string& message);

	static bool LoadFontASCII(ASCIIFont& font, const std::string& filename, float fontSize, unsigned int sampleX = 1, unsigned int sampleY = 1, int padding = 1);
};

constexpr ASCIIFont::ASCIIFont() : pixelHeight(0), scalingFactor(0), ascender(0), descender(0), lineGap(0)
{
	this->characters.fill({});
}

inline const Texture2D& ASCIIFont::GetTexture() const
{
	return this->texture;
}

#endif // FONT_H