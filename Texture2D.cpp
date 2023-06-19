#include "Texture2D.h"
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
#include <iostream>

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
	}
	else
	{
		printf("Error Loading Image '%s': %s\n", filename.c_str(), stbi_failure_reason());
	}
	stbi_image_free((void*) data);
}

void Texture2D::CreateEmpty(std::size_t width, std::size_t height, TextureFormatInternal type, GLint level)
{
	this->CleanUp();
	glGenTextures(1, &this->texture);
	glBindTexture(GL_TEXTURE_2D, this->texture);
	glTexImage2D(GL_TEXTURE_2D, level, (GLenum) type, (GLsizei) width, (GLsizei) height, 0, (GLenum) type, GL_UNSIGNED_BYTE, NULL);
}
