#include "Texture2D.h"
#include "stb_image.h"

// TODO: General texture class thingy

inline constexpr GLenum Texture2D::TextureType()
{
	return GL_TEXTURE_2D;
}

Texture2D::Texture2D() : width(0), height(0), channels(0), data(nullptr), texture(0)
{

}

Texture2D::Texture2D(const std::string& filename) : width(0), height(0), channels(0), data(nullptr), texture(0)
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
		glFinish();
		this->texture = 0;
	}
	if (this->data)
	{
		stbi_image_free(this->data);
		this->data = nullptr;
	}
	this->width = 0;
	this->height = 0;
	this->channels = 0;
}

// TODO: might have to do some gfunky stuff for this given differet types of data but you know
void Texture2D::Load(const std::string& filename)
{
	this->CleanUp();
	this->data = stbi_load(filename.c_str(), &this->width, &this->height, &this->channels, 0);
	glGenTextures(1, &this->texture);
	if (this->data && this->texture)
	{
		glBindTexture(GL_TEXTURE_2D, this->texture);
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1); // Might only be required for jpgs 
		GLenum size = (this->channels == 4) ? GL_RGBA : GL_RGB;
		// TODO: investigate the type of storage types
		glTexImage2D(GL_TEXTURE_2D, 0, size, this->width, this->height, 0, size, GL_UNSIGNED_BYTE, this->data);
		glGenerateMipmap(GL_TEXTURE_2D);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_LINEAR);
	}
	else
	{
		stbi_image_free(this->data);
		this->data = nullptr;
		glDeleteTextures(1, &this->texture);
	}
}

void Texture2D::CreateEmpty(std::size_t width, std::size_t height, GLenum type, GLint level)
{
	this->CleanUp();
	glGenTextures(1, &this->texture);
	glBindTexture(GL_TEXTURE_2D, this->texture);
	glTexImage2D(GL_TEXTURE_2D, level, type, (GLsizei) width, (GLsizei) height, 0, type, GL_UNSIGNED_BYTE, NULL);
}
