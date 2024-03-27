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
#include "util.h"

// TODO: some kind of "base" texture class to reduce clutter
class Texture2D
{
private:
	GLuint texture;
	// Why is channels stored??
	GLsizei width, height, channels;
	GLenum internalFormat;

	inline constexpr GLenum TextureType();
public:
	Texture2D();
	Texture2D(const std::string& filename);
	~Texture2D();

	inline GLuint GetGLTexture() const;

	inline int GetWidth() const;
	inline int GetHeight() const;

	void CleanUp();

	void ApplyInfo(GLuint texture, int width, int height, int channels);

	void MakeAliasOf(Texture2D& other);

	void CopyFrom(Texture2D& other);
	void CopyFrom(Texture2D&& other);
	void CopyFromFramebuffer(const glm::ivec2& size, TextureFormatInternal internalFormat = InternalRGBA, const glm::ivec2& start = glm::ivec2(0, 0));

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
	void CreateEmpty(std::size_t width, std::size_t height, TextureFormatInternal type = InternalRGBA, const glm::vec4& color = glm::vec4(0), GLint level = 0);
	void CreateEmptyWithFilters(std::size_t width, std::size_t height, TextureFormatInternal type = InternalRGBA, const glm::vec4& color = glm::vec4(0), GLint level = 0);

	void FillTexture(const glm::vec4& color, int level = 0);

	template <class T> void Load(const std::vector<T>& data, TextureFormatInternal internal, TextureFormat textureFormat, 
								TextureDataInput dataFormat, std::size_t width, std::size_t height);
	template <class T, std::size_t L> void Load(const std::array<T, L>& data, TextureFormatInternal internal, TextureFormat textureFormat, 
																	TextureDataInput dataFormat, std::size_t width = 0, std::size_t height = 0);
};

inline GLuint Texture2D::GetGLTexture() const
{
	return this->texture;
}

inline int Texture2D::GetWidth() const
{
	return this->width;
}

inline int Texture2D::GetHeight() const
{
	return this->height;
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
	glTextureParameteri(this->texture, GL_TEXTURE_MAG_FILTER, static_cast<GLint>(value));
}

inline void Texture2D::SetMinFilter(TextureMinFilter value) const
{
	if (!(value == MinLinear || value == MinNearest))
		this->GenerateMipmap();
	glTextureParameteri(this->texture, GL_TEXTURE_MIN_FILTER, static_cast<GLint>(value));
}

inline void Texture2D::SetWrapBehaviorS(TextureWrapping value) const
{
	glTextureParameteri(this->texture, GL_TEXTURE_WRAP_S, static_cast<GLint>(value));
}

inline void Texture2D::SetWrapBehaviorT(TextureWrapping value) const
{
	glTextureParameteri(this->texture, GL_TEXTURE_WRAP_T, static_cast<GLint>(value));
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
	glTextureParameterf(this->texture, GL_TEXTURE_MAX_ANISOTROPY, (value > max) ? max : value);
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
		glTexImage2D(GL_TEXTURE_2D, 0, static_cast<GLenum>(internal), static_cast<GLsizei>(width), static_cast<GLsizei>(height), 
			BORDER_PARAMETER, static_cast<GLenum>(textureFormat), static_cast<GLenum>(dataFormat), data.data());
		this->internalFormat = static_cast<GLenum>(internal);
		this->width = static_cast<GLsizei>(width);
		this->height = static_cast<GLsizei>(height);
		this->channels = 3; // TODO: Get a proper calculation on this
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
		glTexImage2D(GL_TEXTURE_2D, 0, static_cast<GLenum>(internal), static_cast<GLsizei>(width), static_cast<GLsizei>(height),
			BORDER_PARAMETER, static_cast<GLenum>(textureFormat), static_cast<GLenum>(dataFormat), data.data());
		this->internalFormat = static_cast<GLenum>(internal);
		this->width = static_cast<GLsizei>(width);
		this->height = static_cast<GLsizei>(height);
		this->channels = 3;
		this->SetFilters();
	}
}


#endif //TEXTURE_2D_H