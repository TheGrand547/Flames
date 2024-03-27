#pragma once
#ifndef FONT_H
#define FONT_H
#include <array>
#include "Buffer.h"
#include "Framebuffer.h"
#include "stbWrangler.h"
#include "Texture2D.h"

namespace Font
{
	void SetFontDirectory(const std::string& directory);
}

// TODO: Settings struct with back/foreground color, position, etc

class ASCIIFont
{
protected:
	std::array<stbtt_packedchar, static_cast<std::size_t>('~' - ' ')> characters;
	float pixelHeight, scalingFactor;
	int ascender, descender, lineGap;
	float lineSkip;

	Texture2D texture; // TODO: Maybe shared texture class or smth?, maybe just a shared pointer who knows
public:
	inline ASCIIFont();

	inline Texture2D& GetTexture();

	// Provide a buffer filled with the position and texture coordinates that with this font's texture can be draw to whatever target
	void RenderToScreen(Buffer<ArrayBuffer>& buffer, float x, float y, const std::string& message);
	void RenderToScreen(Buffer<ArrayBuffer>& buffer, const glm::vec2& coords, const std::string& message);

	// TODO: Maybe a renderbuffer?
	// Standard framebuffer is set to active after this is called
	ColorFrameBuffer Render(const std::string& message, const glm::vec4& textColor = glm::vec4(1), const glm::vec4& backgroundColor = glm::vec4(0));

	// TODO: This is bad
	// Renders directly to the texture
	void Render(Texture2D& texture, float x, float y, const std::string& message);
	void Render(Texture2D& texture, const glm::vec2& coords, const std::string& message);

	static bool LoadFont(ASCIIFont& font, const std::string& filename, float fontSize, unsigned int sampleX = 1, unsigned int sampleY = 1, int padding = 1);
};

inline ASCIIFont::ASCIIFont() : pixelHeight(0), scalingFactor(0), ascender(0), descender(0), lineGap(0), lineSkip(0)
{
	this->characters.fill({});
}

inline Texture2D& ASCIIFont::GetTexture()
{
	return this->texture;
}

#endif // FONT_H