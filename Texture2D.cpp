#include "Texture2D.h"
#include "glmHelp.h"
#include "stbWrangler.h"

// TODO: General texture class thingy

inline constexpr GLenum Texture2D::TextureType()
{
	return GL_TEXTURE_2D;
}

Texture2D::Texture2D() : width(0), height(0), channels(0), texture(0)
{

}

Texture2D::Texture2D(const std::string& filename) : width(0), height(0), channels(0), texture(0)
{
	this->Load(filename);
}

Texture2D::~Texture2D()
{
	this->CleanUp();
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
}

void Texture2D::ApplyInfo(GLuint texture, int width, int height, int channels)
{
	this->CleanUp();
	this->texture = texture;
	this->width = width;
	this->height = height;
	this->channels = channels;
}

void Texture2D::Load(const std::string& filename, TextureFormatInternal internal)
{
	this->CleanUp();
	const unsigned char *data = stbi_load(filename.c_str(), &this->width, &this->height, &this->channels, 0);
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
			internal = (TextureFormatInternal) size;
		}
		glTexImage2D(GL_TEXTURE_2D, 0, (GLenum) internal, this->width, this->height, 0, size, GL_UNSIGNED_BYTE, data);
		this->SetFilters();
	}
	else
	{
		printf("Error Loading Image '%s': %s\n", filename.c_str(), stbi_failure_reason());
	}
	stbi_image_free((void*) data);
}

void Texture2D::CreateEmpty(std::size_t width, std::size_t height, TextureFormatInternal type, const glm::vec4& color, GLint level)
{
	this->CleanUp();
	glGenTextures(1, &this->texture);
	glBindTexture(GL_TEXTURE_2D, this->texture);
	GLenum internalFormat = type, pixelType = type, pixelDataFormat = GL_UNSIGNED_BYTE;
	switch (type)
	{
	case InternalStencil:
		glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_STENCIL_TEXTURE_MODE, GL_STENCIL_INDEX);
		internalFormat = GL_STENCIL_INDEX8;
		pixelType = GL_STENCIL_INDEX;
		break;
	case InternalDepthStencil:
		pixelDataFormat = GL_DEPTH24_STENCIL8;
		pixelDataFormat = GL_UNSIGNED_INT_24_8_EXT;
		break;
	case InternalDepthStencilFloat:
		pixelDataFormat = GL_DEPTH32F_STENCIL8; 
		internalFormat = GL_DEPTH_STENCIL;
		pixelType = GL_DEPTH32F_STENCIL8;
		break;
	case InternalDepth16:
	case InternalDepth24:
	case InternalDepthFloat32:
		pixelType = GL_DEPTH_COMPONENT;
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
	}
	// TODO: store these values or something so things like FillTexture won't throw an annoying error
	glTexImage2D(GL_TEXTURE_2D, level, internalFormat, static_cast<GLsizei>(width), static_cast<GLsizei>(height), 0, pixelType, pixelDataFormat, nullptr);
	glClearTexImage(this->texture, level, pixelType, GL_FLOAT, &color);
	this->width = static_cast<GLsizei>(width);
	this->height = static_cast<GLsizei>(height);
}

void Texture2D::CreateEmptyWithFilters(std::size_t width, std::size_t height, TextureFormatInternal type, const glm::vec4& color, GLint level)
{
	this->CreateEmpty(width, height, type, color, level);
	this->SetFilters();
}

void Texture2D::FillTexture(const glm::vec4& color, int level)
{
	glClearTexImage(this->texture, level, GL_RGBA, GL_FLOAT, &color);
}
