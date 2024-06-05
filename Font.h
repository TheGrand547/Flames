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

struct FontSettings
{
	glm::vec2 position        = glm::vec2(0);
	glm::vec4 textColor       = glm::vec4(1);
	glm::vec4 backgroundColor = glm::vec4(0);
};

class ASCIIFont
{
protected:
	std::array<stbtt_packedchar, static_cast<std::size_t>('~' - ' ')> characters;
	float pixelHeight, scalingFactor;
	int ascender, descender, lineGap, atlasWidth, atlasHeight;
	float lineSkip;

	Texture2D texture;
public:
	inline ASCIIFont() noexcept;
	~ASCIIFont() noexcept;

	inline Texture2D& GetTexture() noexcept;

	void Clear();

	void Load(const std::string& filename, float fontSize, unsigned int sampleX = 1, unsigned int sampleY = 1, int padding = 1);

	// Provide a buffer filled with the position and texture coordinates that with this font's texture can be draw to whatever target
	glm::vec2 GetTextTris(Buffer<ArrayBuffer>& buffer, float x, float y, const std::string& message) const;
	glm::vec2 GetTextTris(Buffer<ArrayBuffer>& buffer, const glm::vec2& coords, const std::string& message) const;

	// Standard framebuffer is set to active after this is called
	ColorFrameBuffer Render(const std::string& message, const glm::vec4& textColor = glm::vec4(1), const glm::vec4& backgroundColor = glm::vec4(0)) const;
	void Render(ColorFrameBuffer& framebuffer, const std::string& message, const glm::vec4& textColor = glm::vec4(1), const glm::vec4& backgroundColor = glm::vec4(0)) const;

	// Renders directly to the texture
	void RenderToTexture(Texture2D& texture, const std::string& message, const glm::vec4& textColor = glm::vec4(1), const glm::vec4& backgroundColor = glm::vec4(0)) const;
	
	// Renders on top of what was already draw to the texture, without resizing
	void RenderOntoTexture(Texture2D& texture, const std::string& message, const glm::vec4& textColor = glm::vec4(1), const glm::vec4& backgroundColor = glm::vec4(0)) const;

	static bool LoadFont(ASCIIFont& font, const std::string& filename, float fontSize, unsigned int sampleX = 1, unsigned int sampleY = 1, int padding = 1, 
		int atlasWidth = 0, int atlasHeight = 0);
};

inline ASCIIFont::ASCIIFont() noexcept : pixelHeight(0), scalingFactor(0), ascender(0), descender(0), lineGap(0), atlasWidth(0), atlasHeight(0), lineSkip(0)
{
	this->characters.fill({});
}

inline Texture2D& ASCIIFont::GetTexture() noexcept
{
	return this->texture;
}

#endif // FONT_H