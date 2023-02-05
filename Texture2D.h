#pragma once
#ifndef TEXTURE_H
#define TEXTURE_H
#include <glew.h>
#include <glm.hpp>
#include <string>

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

	inline void SetMagFilter(TextureMagFilter value) const;
	inline void SetMinFilter(TextureMinFilter value) const;
	inline void SetWrapBehaviorS(TextureWrapping value) const;
	inline void SetWrapBehaviorT(TextureWrapping value) const;
	void Bind(GLuint slot = 0) const;
	void Load(const std::string& filename);
};

#endif TEXTURE_H