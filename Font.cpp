#include "Font.h"
#include <filesystem>
#include <fstream>
#include <glm/ext/matrix_clip_space.hpp>
#include "glUtil.h"
#include "log.h"
#include "Shader.h"
#include "Vertex.h"
#include "VertexArray.h"

static std::string fontVertex = "#version 440 core\nlayout(location = 0)\nin vec2 vPos;layout(location = 1) in vec2 vTex \
;layout(location = 0) out vec2 fTex;uniform mat4 Projection;void main()\
{	gl_Position = Projection * vec4(vPos, 0, 1);	fTex = vTex;}";
static std::string fontFragment = "#version 440 core\nlayout(location = 0) in vec2 fTex;\nout vec4 color;uniform sampler2D fontTexture; \
uniform vec4 colorIn; void main() { float value = texture(fontTexture, fTex).r; if (value == 0) { discard; } color = colorIn * value;}";

static ColorFrameBuffer defaultRenderBuffer;

namespace Font
{
	static Shader shader;
	static VAO vao;

	static void SetupShader()
	{
		shader.CompileEmbedded(fontVertex.c_str(), fontFragment.c_str());
		vao.Generate();
		vao.ArrayFormat<UIVertex>(shader);
	}

	static std::string basePath = "/";

	void SetFontDirectory(const std::string& directory)
	{
		basePath = directory;
	}
}

constexpr std::size_t atlasWidth = 1000; // Magic number of magic numbers
constexpr std::size_t atlasHeight = 700; // Magic number of magic numbers
constexpr int fontBufferStride = 0; // Tightly Packed
constexpr char firstCharInAtlas = ' ';
constexpr char lastCharInAtlas = '~';
constexpr int charsInAtlas = lastCharInAtlas - firstCharInAtlas;
constexpr int fontIndex0 = 0;

ASCIIFont::~ASCIIFont()
{
	this->Clear();
}

void ASCIIFont::Clear()
{
	this->texture.CleanUp();
	this->characters.fill({});
	this->ascender = 0;
	this->lineSkip = 0;
	this->lineGap = 0;
}

void ASCIIFont::Load(const std::string& filename, float fontSize, unsigned int sampleX, unsigned int sampleY, int padding)
{
	ASCIIFont::LoadFont(*this, filename, fontSize, sampleX, sampleY, padding);
}

glm::vec2 ASCIIFont::GetTextTris(Buffer<ArrayBuffer>& buffer, float x, float y, const std::string& message) const
{
	buffer.CleanUp();
	std::vector<UIVertex> results{};
	results.reserve(6 * message.size());
	
	y += this->pixelHeight * 0.8f;
	float originX = x, originY = y;
	stbtt_aligned_quad quad{};
	
	int width = 0, height = static_cast<int>(std::ceil(this->lineSkip));

	for (char letter : message)
	{
		if (letter >= firstCharInAtlas && letter <= lastCharInAtlas)
		{
			// Align to integer because it looks slightly better probably, I don't know
			stbtt_GetPackedQuad(this->characters.data(), atlasWidth, atlasHeight, letter - firstCharInAtlas, &x, &y, &quad, 1);

			width  = std::max( width, static_cast<int>(std::ceil(quad.x1)));
			height = std::max(height, static_cast<int>(std::ceil(quad.y1)));


			results.emplace_back<UIVertex>({ {quad.x0, quad.y0}, {quad.s0, quad.t0} });
			results.emplace_back<UIVertex>({ {quad.x1, quad.y1}, {quad.s1, quad.t1} });
			results.emplace_back<UIVertex>({ {quad.x1, quad.y0}, {quad.s1, quad.t0} });

			results.emplace_back<UIVertex>({ {quad.x0, quad.y0}, {quad.s0, quad.t0} });
			results.emplace_back<UIVertex>({ {quad.x0, quad.y1}, {quad.s0, quad.t1} });
			results.emplace_back<UIVertex>({ {quad.x1, quad.y1}, {quad.s1, quad.t1} });
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
		else
		{
			LogF("Couldn't find character with hex code '%x' in dictionary\n", static_cast<int>(letter));
		}
	}
	buffer.BufferData(results, StaticDraw);
	return glm::vec2(width, height);
}

glm::vec2 ASCIIFont::GetTextTris(Buffer<ArrayBuffer>& buffer, const glm::vec2& coords, const std::string& message) const
{
	return this->GetTextTris(buffer, coords.x, coords.y, message);
}

// TODO: this is a mess man
void ASCIIFont::Render(ColorFrameBuffer& framebuffer, const std::string& message, const glm::vec4& textColor, const glm::vec4& backgroundColor) const
{
	Buffer<ArrayBuffer> buffer;
	glm::ivec2 size = this->GetTextTris(buffer, 0, 0, message);
	framebuffer.GetColor().CreateEmpty(size.x, size.y, InternalRGBA, backgroundColor);
	framebuffer.GetColor().FillTexture(backgroundColor);
	// Don't want artifacting
	framebuffer.GetColor().SetFilters(MinLinear, MagLinear);
	framebuffer.Assemble();
	glm::mat4 projection = glm::ortho<float>(0.f, static_cast<float>(size.x), static_cast<float>(size.y), 0.f);
	framebuffer.Bind();
	glViewport(0, 0, size.x, size.y);

	// TODO: Ensure blending is active
	Font::shader.SetActiveShader();
	Font::shader.SetTextureUnit(std::string("fontTexture"), this->texture, 0);
	Font::shader.SetMat4("Projection", projection);
	Font::shader.SetVec4("colorIn", textColor);
	Font::vao.BindArrayBuffer(buffer);
	Font::shader.DrawArray<DrawType::Triangle>(buffer);
	BindDefaultFrameBuffer();
}

ColorFrameBuffer ASCIIFont::Render(const std::string& message, const glm::vec4& textColor, const glm::vec4& backgroundColor) const
{
	ColorFrameBuffer framebuffer;
	this->Render(framebuffer, message, textColor, backgroundColor);
	return framebuffer;
}

void ASCIIFont::RenderToTexture(Texture2D& texture,  const std::string& message) const
{
	ColorFrameBuffer buffer = this->Render(message);
	buffer.ReadColorIntoTexture(texture);
}


bool ASCIIFont::LoadFont(ASCIIFont& font, const std::string& filename, float fontSize, unsigned int sampleX, unsigned int sampleY, int padding)
{
	Font::SetupShader();
	font.Clear();
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

		// TODO: Pre-calculate the sizes for the types and y'know
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
		int value = stbtt_PackFontRange(&contextual, rawFontData.data(), fontIndex0, STBTT_POINT_SIZE(fontSize), firstCharInAtlas, charsInAtlas, font.characters.data());
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
		return value;
	}
	return false;
}
