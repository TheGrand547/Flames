#pragma once
#ifndef TEXTURE_2D_H
#define TEXTURE_2D_H
#include <glew.h>
#include <glm/glm.hpp>
#include <string>
#include <array>
#include <vector>
#include <math.h>
#include "log.h"
#include "Texture.h"

class Texture2D
{
private:
	GLuint texture;
	int width, height, channels;

	inline constexpr GLenum TextureType();
public:
	Texture2D();
	Texture2D(const std::string& filename);
	~Texture2D();

	inline GLuint GetGLTexture() const;

	void CleanUp();

	inline void SetMagFilter(TextureMagFilter value) const;
	inline void SetMinFilter(TextureMinFilter value) const;
	inline void SetWrapBehaviorS(TextureWrapping value) const;
	inline void SetWrapBehaviorT(TextureWrapping value) const;
	inline void GenerateMipmap() const;
	inline void SetFilters(TextureMinFilter minFilter = MinNearest, TextureMagFilter magFilter = MagNearest, 
		TextureWrapping sWrapping = Repeat, TextureWrapping tWrapping = Repeat) const;
	inline void SetAnisotropy(const float value);

	inline void BindTexture(GLuint slot = 0) const;
	void Load(const std::string& filename, TextureFormatInternal internal = InternalRGBA);
	void CreateEmpty(std::size_t width, std::size_t height, TextureFormatInternal type = InternalRGBA, GLint level = 0);
	template <class T> void Load(const std::vector<T>& data, TextureFormatInternal internal, TextureFormat textureFormat, 
								TextureDataInput dataFormat, std::size_t width, std::size_t height);
	template <class T, std::size_t L> void Load(const std::array<T, L>& data, TextureFormatInternal internal, TextureFormat textureFormat, 
																	TextureDataInput dataFormat, std::size_t width = 0, std::size_t height = 0);
};

inline GLuint Texture2D::GetGLTexture() const
{
	return this->texture;
}

void Texture2D::BindTexture(GLuint slot) const
{
	glBindTextureUnit(slot, this->texture);
}

inline void Texture2D::GenerateMipmap() const
{
	glGenerateTextureMipmap(this->texture);
}

inline void Texture2D::SetMagFilter(TextureMagFilter value) const
{
	glTextureParameteri(this->texture, GL_TEXTURE_MAG_FILTER, (GLint) value);
}

inline void Texture2D::SetMinFilter(TextureMinFilter value) const
{
	if (!(value == MinLinear || value == MinNearest))
		this->GenerateMipmap();
	glTextureParameteri(this->texture, GL_TEXTURE_MIN_FILTER, (GLint) value);
}

inline void Texture2D::SetWrapBehaviorS(TextureWrapping value) const
{
	glTextureParameteri(this->texture, GL_TEXTURE_WRAP_S, (GLint) value);
}

inline void Texture2D::SetWrapBehaviorT(TextureWrapping value) const
{
	glTextureParameteri(this->texture, GL_TEXTURE_WRAP_T, (GLint) value);
}

inline void Texture2D::SetFilters(TextureMinFilter minFilter, TextureMagFilter magFilter, 
									TextureWrapping sWrapping, TextureWrapping tWrapping) const
{
	this->SetMinFilter(minFilter);
	this->SetMagFilter(magFilter);
	this->SetWrapBehaviorS(sWrapping);
	this->SetWrapBehaviorT(tWrapping);
}

inline void Texture2D::SetAnisotropy(const float value)
{
	float max;
	glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY, &max);
	glTextureParameterf(this->texture, GL_ARB_texture_filter_anisotropic, (value > max) ? max : value);
}

template<class T> inline void Texture2D::Load(const std::vector<T>& data, TextureFormatInternal internal, TextureFormat textureFormat,
	TextureDataInput dataFormat, std::size_t width, std::size_t height)
{
	this->CleanUp();
	glGenTextures(1, &this->texture);
	if (this->texture)
	{
		glBindTexture(GL_TEXTURE_2D, this->texture);
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		glTexImage2D(GL_TEXTURE_2D, 0, (GLenum) internal, (GLsizei) width, (GLsizei) height, 0, (GLenum) textureFormat, (GLenum) dataFormat, data.data());
		this->SetFilters();
	}
}

template<class T, std::size_t L> inline void Texture2D::Load(const std::array<T, L>& data, TextureFormatInternal internal, TextureFormat textureFormat,
																		TextureDataInput dataFormat, std::size_t width, std::size_t height)
{
	this->CleanUp();
	if (!width && !height)
	{
		width = height = (std::size_t)sqrt(L);
		if (width * height != L)
		{
			LogF("Width(%zu) and Height(%zu) do not equal the proper size(%zu)\n", width, height, L);
			return;
		}
	}
	glGenTextures(1, &this->texture);
	if (this->texture)
	{
		glBindTexture(GL_TEXTURE_2D, this->texture);
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		glTexImage2D(GL_TEXTURE_2D, 0, (GLenum) internal, (GLsizei)width, (GLsizei)height, 0, (GLenum)textureFormat, (GLenum)dataFormat, data.data());
		this->SetFilters();
	}
}


#endif //TEXTURE_2D_H