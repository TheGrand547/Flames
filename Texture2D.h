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

struct FilterStruct
{
	TextureMinFilter minFilter = MinNearest;
	TextureMagFilter magFilter = MagNearest;
	TextureWrapping sWrapping = Repeat;
	TextureWrapping tWrapping = Repeat;
};

// TODO: some kind of "base" texture class to reduce clutter
class Texture2D
{
private:
	GLuint texture;
	// Why is channels stored??
	GLsizei width, height, channels;
	GLenum internalFormat;
	bool isTextureView;

	inline constexpr GLenum TextureType();
public:
	Texture2D();
	Texture2D(const std::string& filename);
	~Texture2D();

	Texture2D& operator=(Texture2D&& left) noexcept;

	inline GLuint GetGLTexture() const noexcept;

	inline int GetWidth() const noexcept;
	inline int GetHeight() const noexcept;
	inline glm::ivec2 GetSize() const noexcept;

	inline glm::vec2 GetTextureCoordinates(const glm::ivec2& pos) const noexcept;
	inline glm::vec2 GetTextureCoordinates(int x, int y) const noexcept;

	void CleanUp();

	void ApplyInfo(GLuint texture, int width, int height, int channels);

	void MakeAliasOf(Texture2D& other);

	void CopyFrom(Texture2D& other);
	void CopyFrom(Texture2D&& other);
	void CopyFromFramebuffer(const glm::ivec2& size, TextureFormatInternal internalFormat = InternalRGBA8, const glm::ivec2& start = glm::ivec2(0, 0));

	inline void SetMagFilter(TextureMagFilter value) const;
	inline void SetMinFilter(TextureMinFilter value) const;
	inline void SetWrapBehaviorS(TextureWrapping value) const;
	inline void SetWrapBehaviorT(TextureWrapping value) const;
	inline void GenerateMipmap() const;
	inline void SetFilters(TextureMinFilter minFilter = MinNearest, TextureMagFilter magFilter = MagNearest, 
		TextureWrapping sWrapping = Repeat, TextureWrapping tWrapping = Repeat) const;
	inline void SetFilters(const FilterStruct& filters) const;
	inline void SetAnisotropy(const float value) const;

	inline void BindTexture(GLuint slot = 0) const;
	void Load(const std::string& filename, TextureFormatInternal internal = InternalRGBA8);
	void CreateEmpty(std::size_t width, std::size_t height, TextureFormatInternal type = InternalRGBA8, const glm::vec4& color = glm::vec4(0), GLint level = 0);
	inline void CreateEmpty(glm::ivec2 size, TextureFormatInternal type = InternalRGBA8, const glm::vec4& color = glm::vec4(0), GLint level = 0);
	void CreateEmptyWithFilters(std::size_t width, std::size_t height, TextureFormatInternal type = InternalRGBA8, FilterStruct filters = {},
		const glm::vec4& color = glm::vec4(0), GLint level = 0);
	inline void CreateEmptyWithFilters(glm::ivec2 size, TextureFormatInternal type = InternalRGBA8, FilterStruct filters = {}, 
		const glm::vec4& color = glm::vec4(0), GLint level = 0);

	void FillTexture(const glm::vec4& color, int level = 0) const;

	template <class T> void Load(const std::vector<T>& data, TextureFormatInternal internal, TextureFormat textureFormat, 
								TextureDataInput dataFormat, std::size_t width, std::size_t height);
	template <class T, std::size_t L> void Load(const std::array<T, L>& data, TextureFormatInternal internal, TextureFormat textureFormat, 
																	TextureDataInput dataFormat, std::size_t width = 0, std::size_t height = 0);
};

inline GLuint Texture2D::GetGLTexture() const noexcept
{
	return this->texture;
}

inline int Texture2D::GetWidth() const noexcept
{
	return this->width;
}

inline int Texture2D::GetHeight() const noexcept
{
	return this->height;
}

inline glm::ivec2 Texture2D::GetSize() const noexcept
{
	return glm::ivec2(this->width, this->height);
}

inline glm::vec2 Texture2D::GetTextureCoordinates(const glm::ivec2& pos) const noexcept
{
	return glm::vec2(this->width, this->height) / glm::vec2(pos);
}

inline glm::vec2 Texture2D::GetTextureCoordinates(int x, int y) const noexcept
{
	return this->GetTextureCoordinates(glm::ivec2(x, y));
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

inline void Texture2D::SetFilters(const FilterStruct& filters) const
{
	this->SetMinFilter(filters.minFilter);
	this->SetMagFilter(filters.magFilter);
	this->SetWrapBehaviorS(filters.sWrapping);
	this->SetWrapBehaviorT(filters.tWrapping);
}

inline void Texture2D::SetAnisotropy(const float value) const
{
	float max;
	glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY, &max);
	glTextureParameterf(this->texture, GL_TEXTURE_MAX_ANISOTROPY, glm::min(max, value));
}

inline void Texture2D::CreateEmpty(glm::ivec2 size, TextureFormatInternal type, const glm::vec4& color, GLint level)
{
	this->CreateEmpty(static_cast<std::size_t>(size.x), static_cast<std::size_t>(size.y), type, color, level);
}

inline void Texture2D::CreateEmptyWithFilters(glm::ivec2 size, TextureFormatInternal type, FilterStruct filters, const glm::vec4& color, GLint level)
{
	this->CreateEmptyWithFilters(static_cast<std::size_t>(size.x), static_cast<std::size_t>(size.y), type, filters, color, level);
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
		this->channels = Texture::GetColorChannels(internal);
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
		this->channels = Texture::GetColorChannels(internal);
		this->SetFilters();
	}
}
#endif //TEXTURE_2D_H