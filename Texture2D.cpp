#include "Texture2D.h"
#include "stb_image.h"
// TODO: Move all stb header implementations to a single file
/*
#pragma warning (push)
#pragma warning (disable: 26451 26453 6001 6262 26819)
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#pragma warning (pop)
*/


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
	if (this->texture)
	{
		glDeleteTextures(1, &this->texture);
	}
	if (this->data)
	{
		stbi_image_free(this->data);
		this->data = nullptr;
	}
}

inline void Texture2D::SetMagFilter(TextureMagFilter value) const
{
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, (GLint) value);
}

inline void Texture2D::SetMinFilter(TextureMinFilter value) const
{
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, (GLint)value);
}

inline void Texture2D::SetWrapBehaviorS(TextureWrapping value) const
{
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, (GLint) value);
}

inline void Texture2D::SetWrapBehaviorT(TextureWrapping value) const
{
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, (GLint)value);
}

void Texture2D::Bind(GLuint slot) const
{
	glActiveTexture(GL_TEXTURE0 + slot);
	glBindTexture(GL_TEXTURE_2D, this->texture);
}
// TODO: might have to do some gfunky stuff for this given differet types of data but you know
void Texture2D::Load(const std::string& filename)
{
	// TODO: write cleanup function
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
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
	}
	else
	{
		stbi_image_free(this->data);
		this->data = nullptr;
		glDeleteTextures(1, &this->texture);
	}
}
