#pragma once
#ifndef TEXTURE_H
#define TEXTURE_H
#include <glew.h>
#include <glm/glm.hpp>
#include <string>
#include <array>
#include <vector>
#include <math.h>

enum TextureMagFilter
{
	MagNearest = GL_NEAREST,
	MagLinear  = GL_LINEAR,
};

enum TextureMinFilter
{
	MinNearest     = GL_NEAREST,
	MinLinear      = GL_LINEAR,
	// First part specifies the texture sampling mode, second the mipmap sampling mode
	NearestNearest = GL_NEAREST_MIPMAP_NEAREST,
	NearestLinear  = GL_NEAREST_MIPMAP_LINEAR,
	LinearNearest  = GL_LINEAR_MIPMAP_NEAREST,
	LinearLinear   = GL_LINEAR_MIPMAP_LINEAR
};

enum TextureWrapping
{
	EdgeClamp         = GL_CLAMP_TO_EDGE, 
	BorderClamp       = GL_CLAMP_TO_BORDER, 
	MirroredRepeat    = GL_MIRRORED_REPEAT, 
	Repeat            = GL_REPEAT, 
	MirroredEdgeClamp = GL_MIRROR_CLAMP_TO_EDGE
};

class Texture2D
{
private:
	GLuint texture;
	int width, height, channels;
	unsigned char* data;

	inline constexpr GLenum TextureType();
public:
	Texture2D();
	Texture2D(const std::string& filename);
	~Texture2D();

	void CleanUp();

	inline void SetMagFilter(TextureMagFilter value) const;
	inline void SetMinFilter(TextureMinFilter value) const;
	inline void SetWrapBehaviorS(TextureWrapping value) const;
	inline void SetWrapBehaviorT(TextureWrapping value) const;
	inline void GenerateMipmap() const;
	inline void SetFilters(TextureMinFilter minFilter, TextureMagFilter magFilter, 
		TextureWrapping sWrapping, TextureWrapping tWrapping) const;

	void Bind(GLuint slot = 0) const;
	void Load(const std::string& filename);
	template <class T> void Load(const std::vector<T>& data);
	template <class T, GLenum type = GL_FLOAT, std::size_t L> void Load(const std::array<T, L>& data, std::size_t width = 0, std::size_t height = 0);
};

inline void Texture2D::GenerateMipmap() const
{
	glGenerateMipmap(this->texture);
}

inline void Texture2D::SetMagFilter(TextureMagFilter value) const
{
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, (GLint)value);
}

inline void Texture2D::SetMinFilter(TextureMinFilter value) const
{
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, (GLint)value);
}

inline void Texture2D::SetWrapBehaviorS(TextureWrapping value) const
{
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, (GLint)value);
}

inline void Texture2D::SetWrapBehaviorT(TextureWrapping value) const
{
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, (GLint)value);
}

inline void Texture2D::SetFilters(TextureMinFilter minFilter, TextureMagFilter magFilter, 
									TextureWrapping sWrapping, TextureWrapping tWrapping) const
{
	this->SetMinFilter(minFilter);
	this->SetMagFilter(magFilter);
	this->SetWrapBehaviorS(sWrapping);
	this->SetWrapBehaviorT(tWrapping);
}

template<class T> inline void Texture2D::Load(const std::vector<T>& data)
{
	// BAD
}

template<class T, GLenum type, std::size_t L> inline void Texture2D::Load(const std::array<T, L>& data, std::size_t width, std::size_t height)
{
	this->CleanUp();
	glGenTextures(1, &this->texture);
	if (!width && !height)
	{
		width = height = (std::size_t) sqrt(L);
	}
	if (this->texture)
	{
		glBindTexture(GL_TEXTURE_2D, this->texture);
		// TODO: Don't just assume a single channel
		// TODO: Think this is wrong and stuff but i'm not sure
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, (GLsizei) width, (GLsizei) height, 0, GL_RED, type, data.data());
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	}
}


#endif //TEXTURE_H