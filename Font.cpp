#include "Font.h"
#include <filesystem>
#include <fstream>
#include "log.h"

static std::string fontBasePath = "/";

void Font::SetFontDirectory(const std::string& directory)
{
	fontBasePath = directory;
}

constexpr std::size_t atlasWidth = 1000; // Magic number of magic numbers
constexpr std::size_t atlasHeight = 1000; // Magic number of magic numbers
constexpr int fontBufferStride = 0; // Tightly Packed

bool Font::LoadFont(Font& font, const std::string& filename, float fontSize, unsigned int sampleX, unsigned int sampleY, int padding)
{
	std::filesystem::path fontFile(fontBasePath + filename);
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
		// TODO: figure out what the scratchSpace is for
		std::vector<unsigned char> scratchSpace{};
		scratchSpace.reserve(atlasWidth * atlasHeight);

		stbtt_fontinfo information{};
		stbtt_InitFont(&information, rawFontData.data(), 0);
		int a = 0, d = 0, lg = 0;
		stbtt_GetFontVMetrics(&information, &a, &d, &lg);
		font.scalingFactor = fontSize / (a - d);

		stbtt_pack_context contextual{};

		// Why is this nullptr
		stbtt_PackBegin(&contextual, scratchSpace.data(), atlasWidth, atlasHeight, fontBufferStride, padding, nullptr);
		stbtt_PackSetOversampling(&contextual, sampleX, sampleY);
		stbtt_PackFontRange(&contextual, rawFontData.data(), 0, STBTT_POINT_SIZE(fontSize), ' ', 96, font.characters.data());
		stbtt_PackEnd(&contextual);

		rawFontData.clear();
		scratchSpace.clear();
		input.close();
	}
	return false;
}
