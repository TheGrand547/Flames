#include "Texture2D.h"
#include "glmHelp.h"
#include "stbWrangler.h"
#include "util.h"
#include <glm/gtc/type_ptr.hpp>

// TODO: General texture class thingy

inline constexpr GLenum Texture2D::TextureType()
{
	return GL_TEXTURE_2D;
}

Texture2D::Texture2D() : width(0), height(0), channels(0), texture(0), internalFormat(0), isTextureView(false)
{

}

Texture2D::Texture2D(const std::string& filename, const FilterStruct& filters) : width(0), height(0), channels(0), texture(0), internalFormat(0), isTextureView(false)
{
	this->Load(filename);
	this->SetFilters(filters);
}

Texture2D::~Texture2D()
{
	this->CleanUp();
}

Texture2D& Texture2D::operator=(Texture2D&& left) noexcept
{
	this->CleanUp();
	std::swap(this->texture, left.texture);
	std::swap(this->width, left.width);
	std::swap(this->height, left.height);
	std::swap(this->channels, left.channels);
	std::swap(this->internalFormat, left.internalFormat);
	std::swap(this->isTextureView, left.isTextureView);
	return *this;
}

void Texture2D::CleanUp()
{
	if (this->texture)
	{
		glDeleteTextures(1, &this->texture);
		this->texture = 0;
	}
	this->width = 0;
	this->height = 0;
	this->channels = 0;
	this->internalFormat = 0;
	this->isTextureView = false;
}

void Texture2D::ApplyInfo(GLuint texture, int width, int height, int channels)
{
	this->CleanUp();
	this->texture = texture;
	this->width = width;
	this->height = height;
	this->channels = channels;
	this->internalFormat = 0;
}

void Texture2D::MakeAliasOf(const Texture2D& other)
{
	this->CleanUp();
	glGenTextures(1, &this->texture);
	glTextureView(this->texture, GL_TEXTURE_2D, other.texture, other.internalFormat, 0, 1, 0, 1);
	this->width = other.width;
	this->height = other.height;
	this->channels = other.channels;
	this->internalFormat = other.internalFormat;
	this->isTextureView = true;
}

void Texture2D::CopyFrom(Texture2D& other)
{
	Log("TODO");
}

void Texture2D::CopyFrom(Texture2D&& other)
{
	*this = std::forward<Texture2D&&>(other);
}

void Texture2D::CopyFromFramebuffer(const glm::ivec2& size, TextureFormatInternal internalFormat, const glm::ivec2& start)
{
	CheckError();
	this->CreateEmptyWithFilters(size.x, size.y, internalFormat, {}, glm::vec4(0.5));
	CheckError();
	glBindTexture(GL_TEXTURE_2D, this->texture);
	CheckError();
	glCopyTexImage2D(GL_TEXTURE_2D, 0, static_cast<GLenum>(internalFormat), start.x, start.y, size.x, size.y, BORDER_PARAMETER);
	CheckError();
	this->width = size.x;
	this->height = size.y;
	this->internalFormat = static_cast<GLenum>(internalFormat);
	this->channels = Texture::GetColorChannels(internalFormat); 
}

void Texture2D::Load(const std::string& filename, TextureFormatInternal internal)
{
	this->CleanUp();
	const unsigned char *data = stbi_load((Texture::GetBasePath() + filename).c_str(), &this->width, &this->height, &this->channels, 0);
	if (data)
	{
		glGenTextures(1, &this->texture);
		if (!this->texture)
			return;
		glBindTexture(GL_TEXTURE_2D, this->texture);
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1); // All stbi_loaded images are continuous in memory
		GLenum size;
		switch (this->channels)
		{
			case 1: size = FormatRed; break;
			case 2: size = FormatRedGreen; break;
			case 3: size = FormatRGB; break;
			default: size = FormatRGBA; break;
		}
		if (internal == InternalUnspecified)
		{	
			internal = static_cast<TextureFormatInternal>(size);
		}
		glTexImage2D(GL_TEXTURE_2D, 0, static_cast<GLenum>(internal), this->width, this->height, BORDER_PARAMETER, size, GL_UNSIGNED_BYTE, data);
		this->internalFormat = static_cast<GLenum>(internal);
		this->SetFilters();
	}
	else
	{
		printf("Error Loading Image '%s%s': %s\n", Texture::GetBasePath().c_str(), filename.c_str(), stbi_failure_reason());
	}
	stbi_image_free(std::bit_cast<void*>(data));
}

void Texture2D::CreateEmpty(std::size_t width, std::size_t height, TextureFormatInternal type, const glm::vec4& color, GLint level)
{
	this->CleanUp();
	glGenTextures(1, &this->texture);
	glBindTexture(GL_TEXTURE_2D, this->texture);
	GLenum internalFormat = type, pixelType = GL_RGBA;

	switch (type)
	{
	case InternalStencil:
		glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_STENCIL_TEXTURE_MODE, GL_STENCIL_INDEX);
		internalFormat = GL_STENCIL_INDEX8;
		pixelType = GL_STENCIL_INDEX;
		break;
	case InternalDepthStencil:
		internalFormat = GL_DEPTH24_STENCIL8;
		pixelType = GL_DEPTH24_STENCIL8;
		break;
	case InternalDepthStencilFloat:
		internalFormat = GL_DEPTH32F_STENCIL8;
		pixelType = GL_DEPTH32F_STENCIL8;
		break;
	case InternalDepth:
	case InternalDepth16:
	case InternalDepth24:
	case InternalDepthFloat32:
		pixelType = GL_DEPTH_COMPONENT;
		internalFormat = GL_DEPTH_COMPONENT32F;
		break;
	case InternalFloatRed16:
	case InternalFloatRedGreen16: 
	case InternalFloatRGB16:
	case InternalFloatRGBA16:
	case InternalFloatRed32 :
	case InternalFloatRedGreen32:
	case InternalFloatRGB32:
	case InternalFloatRGBA32:
	case InternalFloatR11G11B10:
	case InternalFloatShared5RGB9:
	case InternalFloatBPTCRGB:
	case InternalUnsignedFloatBPTCRGB:
		pixelType = GL_RED; // Can't have floating point pixelType
		break;
	case InternalRed:
		pixelType = GL_RED;
		internalFormat = GL_R8;
		break;
	case InternalRedGreen:
		pixelType = GL_RG;
		internalFormat = GL_RG8;
		break;
	case InternalRGB:
		pixelType = GL_RGB;
		internalFormat = GL_RGB8;
		break;
	case InternalRGBA:
		pixelType = GL_RGBA;
		internalFormat = GL_RGBA8;
		break;
	}
	this->width = static_cast<GLsizei>(width);
	this->height = static_cast<GLsizei>(height);

	glTextureStorage2D(this->texture, level + 1, internalFormat, this->width, this->height);
	glm::u8vec4 clamped{};
	for (int i = 0; i < 4; i++)
	{
		clamped[i] = static_cast<unsigned char>(color[i] * 255);
	}
	// A mess, but it finally doesn't yet at me every time, yay!
	glClearTexImage(this->texture, level, (type == GL_DEPTH_STENCIL) ? GL_DEPTH_STENCIL : pixelType, 
		(type == GL_DEPTH_STENCIL) ? GL_UNSIGNED_INT_24_8_EXT : GL_FLOAT, glm::value_ptr(color));
	//glClearTexImage(this->texture, level, pixelType, GL_UNSIGNED_BYTE, glm::value_ptr(clamped));
	this->internalFormat = internalFormat;
	this->channels = Texture::GetColorChannels(type);


	// TODO: Texture fill type so clear tex image will behave, stencil/depth/depth+stencil get their own and everything else rgba
}

void Texture2D::CreateEmptyWithFilters(std::size_t width, std::size_t height, TextureFormatInternal type, FilterStruct filters, const glm::vec4& color, GLint level)
{
	// TODO: FIX THIS HACK
	this->CleanUp();
	glGenTextures(1, &this->texture);
	glBindTexture(GL_TEXTURE_2D, this->texture);
	this->SetFilters(filters);
	GLenum internalFormat = type, pixelType = GL_RGBA;
	// TODO: Remove these I don't think they're necessary
	switch (type)
	{
	case InternalStencil:
		glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_STENCIL_TEXTURE_MODE, GL_STENCIL_INDEX);
		internalFormat = GL_STENCIL_INDEX8;
		pixelType = GL_STENCIL_INDEX;
		break;
	case InternalDepthStencil:
		pixelType = GL_DEPTH_STENCIL;
		break;
	case InternalDepthStencilFloat:
		internalFormat = GL_DEPTH_STENCIL;
		pixelType = GL_DEPTH_STENCIL;
		break;
	case InternalDepth:
	case InternalDepth16:
	case InternalDepth24:
	case InternalDepthFloat32:
		pixelType = GL_DEPTH_COMPONENT;
		internalFormat = GL_DEPTH_COMPONENT32F;
		break;
	case InternalFloatRed16:
	case InternalFloatRedGreen16:
	case InternalFloatRGB16:
	case InternalFloatRGBA16:
	case InternalFloatRed32:
	case InternalFloatRedGreen32:
	case InternalFloatRGB32:
	case InternalFloatRGBA32:
	case InternalFloatR11G11B10:
	case InternalFloatShared5RGB9:
	case InternalFloatBPTCRGB:
	case InternalUnsignedFloatBPTCRGB:
		pixelType = GL_RED; // Can't have floating point pixelType
		break;
	case InternalRed:
		pixelType = GL_RED;
		internalFormat = GL_R8;
		break;
	case InternalRedGreen:
		pixelType = GL_RG;
		internalFormat = GL_RG8;
		break;
	case InternalRGB:
		pixelType = GL_RGB;
		internalFormat = GL_RGB8;
		break;
	case InternalRGBA:
		pixelType = GL_RGBA;
		internalFormat = GL_RGBA8;
		break;
	}
	this->width = static_cast<GLsizei>(width);
	this->height = static_cast<GLsizei>(height);

	glTextureStorage2D(this->texture, level + 1, internalFormat, this->width, this->height);
	glClearTexImage(this->texture, level, pixelType, GL_FLOAT, glm::value_ptr(color));
	this->internalFormat = internalFormat;
	this->channels = Texture::GetColorChannels(type);
	if (!(filters.minFilter == MinLinear || filters.minFilter == MinNearest))
		this->GenerateMipmap();
}

void Texture2D::FillTexture(const glm::vec4& color, int level) const
{
	glClearTexImage(this->texture, level, GL_RGBA, GL_FLOAT, glm::value_ptr(color));
}
