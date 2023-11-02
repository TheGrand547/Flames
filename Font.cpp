#include "Font.h"
#include <filesystem>
#include <fstream>
#include "log.h"
#include "Vertex.h"

namespace Font
{
	static std::string basePath = "/";

	void SetFontDirectory(const std::string& directory)
	{
		basePath = directory;
	}
}

constexpr std::size_t atlasWidth = 1000; // Magic number of magic numbers
constexpr std::size_t atlasHeight = 500; // Magic number of magic numbers
constexpr int fontBufferStride = 0; // Tightly Packed
constexpr char firstCharInAtlas = ' ';
constexpr char lastCharInAtlas = '~';
constexpr int charsInAtlas = lastCharInAtlas - firstCharInAtlas;
constexpr int fontIndex0 = 0;

void ASCIIFont::Render(Buffer<ArrayBuffer>& buffer, float x, float y, const std::string& message)
{
	// TODO: Get the screen size
	const float screenWidth = 1000.f;
	const float screenHeight = 1000.f;

	buffer.CleanUp();
	std::vector<UIVertex> results{};
	results.reserve(6 * message.size());
	float originX = x, originY = y;
	stbtt_aligned_quad quad{};
	for (char letter : message)
	{
		if (letter >= firstCharInAtlas && letter <= lastCharInAtlas)
		{
			// I don't know why the 'align to integer' param is 1
			stbtt_GetPackedQuad(this->characters.data(), atlasWidth, atlasHeight, letter - firstCharInAtlas, &x, &y, &quad, 1);

			results.push_back({ {quad.x0, -quad.y1}, {quad.s0, quad.t1} });
			results.push_back({ {quad.x1, -quad.y0}, {quad.s1, quad.t0} });
			results.push_back({ {quad.x0, -quad.y0}, {quad.s0, quad.t0} });

			results.push_back({ {quad.x0, -quad.y1}, {quad.s0, quad.t1} });
			results.push_back({ {quad.x1, -quad.y1}, {quad.s1, quad.t1} });
			results.push_back({ {quad.x1, -quad.y0}, {quad.s1, quad.t0} });
		}
		else if (letter == '\n')
		{
			x = originX;
			y = originY + this->lineSkip;
			originY = y;
		}
		else if (letter == '\t')
		{
			// SLOPPY
			stbtt_GetPackedQuad(this->characters.data(), atlasWidth, atlasHeight, ' ', &x, &y, &quad, 1);
			stbtt_GetPackedQuad(this->characters.data(), atlasWidth, atlasHeight, ' ', &x, &y, &quad, 1);
			stbtt_GetPackedQuad(this->characters.data(), atlasWidth, atlasHeight, ' ', &x, &y, &quad, 1);
			stbtt_GetPackedQuad(this->characters.data(), atlasWidth, atlasHeight, ' ', &x, &y, &quad, 1);
		}
	}
	for (auto& s : results)
	{
		s.position.y += 2 * screenHeight - this->lineSkip;
		s.position /= glm::vec2(screenWidth, screenHeight);
		s.position -= 1.f;
	}
	buffer.Generate();
	buffer.BufferData(results, StaticDraw);
}

void ASCIIFont::Render(Buffer<ArrayBuffer>& buffer, const glm::vec2& coords, const std::string& message)
{
	this->Render(buffer, coords.x, coords.y, message);
}

void ASCIIFont::Render(Texture2D& texture, float x, float y, const std::string& message)
{
	Log("TODO DUMBASS");
}

void ASCIIFont::Render(Texture2D& texture, const glm::vec2& coords, const std::string& message)
{
	this->Render(texture, coords.x, coords.y, message);
}


bool ASCIIFont::LoadFont(ASCIIFont& font, const std::string& filename, float fontSize, unsigned int sampleX, unsigned int sampleY, int padding)
{
	std::filesystem::path fontFile(Font::basePath + "/" + filename);
	if (!std::filesystem::exists(fontFile))
	{
		LogF("Unable to load font file '%s'\n", fontFile.string().c_str());
		return false;
	}
	font.characters.fill({});
	font.pixelHeight = fontSize;
	std::ifstream input{ fontFile, std::ios::binary };
	if (input.good())
	{
		// From stackoverflow, kinda cringe tbh but who cares(me)
		std::streampos fileSize = std::filesystem::file_size(fontFile);
		std::vector<unsigned char> rawFontData{};
		rawFontData.reserve(fileSize);
		input.read(std::bit_cast<char*>(rawFontData.data()), fileSize);

		stbtt_fontinfo information{};
		stbtt_InitFont(&information, rawFontData.data(), fontIndex0);
		stbtt_GetFontVMetrics(&information, &font.ascender, &font.descender, &font.lineGap);
		font.scalingFactor = fontSize / (font.ascender - font.descender);
		font.lineSkip = (font.ascender - font.descender + font.lineGap) * font.scalingFactor;

		// TODO: Dynamically resize the atlas for better memory layout
		int x0, y0, x1, y1;
		stbtt_GetFontBoundingBox(&information, &x0, &y0, &x1, &y1);
		float boundingWidth = (x1 - x0) * font.scalingFactor * sampleX + 1;
		float boundingHeight = (y1 - y0) * font.scalingFactor * sampleY + 1;

		std::vector<unsigned char> scratchSpace{}; // Has to be the same size as the buffer
		scratchSpace.reserve(atlasWidth * atlasHeight);
		stbtt_pack_context contextual{};
		// Why is this nullptr
		stbtt_PackBegin(&contextual, scratchSpace.data(), atlasWidth, atlasHeight, fontBufferStride, padding, nullptr);
		stbtt_PackSetOversampling(&contextual, sampleX, sampleY);
		stbtt_PackSetSkipMissingCodepoints(&contextual, true);
		stbtt_PackFontRange(&contextual, rawFontData.data(), fontIndex0, STBTT_POINT_SIZE(fontSize), firstCharInAtlas, charsInAtlas, font.characters.data());
		stbtt_PackEnd(&contextual);

		font.texture.CleanUp();
		GLuint texture;
		glGenTextures(1, &texture);
		glBindTexture(GL_TEXTURE_2D, texture);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, atlasWidth, atlasHeight, 0, GL_RED, GL_UNSIGNED_BYTE, scratchSpace.data());
		font.texture.ApplyInfo(texture, atlasWidth, atlasHeight, 1);
		font.texture.SetFilters(MinLinear, MagLinear, Repeat, Repeat);


		rawFontData.clear();
		scratchSpace.clear();
		input.close();
		return true;
	}
	return false;
}
