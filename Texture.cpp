#include "Texture.h"

#pragma warning (push)
#pragma warning (disable: 26451 26453 6001 6262 26819)
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#pragma warning (pop)


Texture::Texture() : width(0), height(0), channels(0), data(nullptr), texture(0), type(0)
{

}

Texture::Texture(const std::string& filename, GLenum type) : width(0), height(0), channels(0), data(nullptr), texture(0), type(0)
{
	this->Load(filename, type);
}

Texture::~Texture()
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

void Texture::Bind(GLuint slot) const
{
	glActiveTexture(GL_TEXTURE0 + slot);
	//glBindTexture(this->type, this->texture);
	glBindTexture(GL_TEXTURE_2D, this->texture);
}
// TODO: might have to do some gfunky stuff for this given differet types of data but you know
void Texture::Load(const std::string& filename, GLenum type)
{
	// TODO: write cleanup function
	this->data = stbi_load(filename.c_str(), &this->width, &this->height, &this->channels, 0);
	glGenTextures(1, &this->texture);
	if (this->data && this->texture)
	{
		this->type = GL_TEXTURE_2D;
		glBindTexture(this->type, this->texture);
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
