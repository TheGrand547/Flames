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

enum TextureFormat
{
	FormatRed          = GL_RED,
	FormatRedGreen     = GL_RG,
	FormatRGB          = GL_RGB,
	FormatBGR          = GL_BGR,
	FormatRGBA         = GL_RGBA,
	FormatBGRA         = GL_BGRA,
	FormatDepth        = GL_DEPTH_COMPONENT,
	FormatDepthStencil = GL_DEPTH_STENCIL
};

enum TextureFormatInternal
{
	InternalRed                  = GL_RED,
	InternalRedGreen             = GL_RG,
	InternalRGB                  = GL_RGB,
	InternalRGBA                 = GL_RGBA,
	InternalDepth                = GL_DEPTH,
	InternalDepthStencil         = GL_DEPTH_STENCIL,
	InternalRed8                 = GL_R8,
	InternalSignedRed8           = GL_R8_SNORM,
	InternalRed16                = GL_R16,
	InternalSignedRed16          = GL_R16_SNORM,
	InternalRedGreen16           = GL_RG16,
	InternalSignedRedGreen16     = GL_RG16_SNORM,
	InternalRGB332               = GL_R3_G3_B2,
	InternalRGB4                 = GL_RGB4,
	InternalRGB5                 = GL_RGB5,
	InternalRGB8                 = GL_RGB8,
	InternalSignedRGB8           = GL_RGB8_SNORM,
	InternalRGB10                = GL_RGB10,
	InternalRGB12                = GL_RGBA12,
	InternalRGB16                = GL_RGBA16,
	InternalSignedRGB16          = GL_RGB16_SNORM,
	InternalRGBA2                = GL_RGBA2,
	InternalRGBA4                = GL_RGBA4,
	InternalRGB5A1               = GL_RGB5_A1,
	InternalRGBA8                = GL_RGBA8,
	InternalSignedRGBA8          = GL_RGBA8_SNORM,
	InternalRGB10A2              = GL_RGB10_A2,
	InternalUnsignedIntRGB10A2   = GL_RGB10_A2UI,
	InternalRGBA12               = GL_RGBA12,
	InternalRGBA16               = GL_RGBA16,
	InternalSRGB8                = GL_SRGB8,
	InternalSRGBA8               = GL_SRGB8_ALPHA8,
	InternalFloatRed16           = GL_R16F,
	InternalFloatRedGreen16      = GL_RG16F,
	InternalFloatRGB16           = GL_RGB16F,
	InternalFloatRGBA16          = GL_RGBA16F,
	InternalFloatRed32           = GL_R32F,
	InternalFloatRedGreen32      = GL_RG32F,
	InternalFloatRGB32           = GL_RGB32F,
	InternalFloatRGBA32          = GL_RGBA32F,
	InternalFloatR11G11B10       = GL_R11F_G11F_B10F,
	InternalFloatShared5RGB9     = GL_RGB9_E5,
	InternalIntRed8              = GL_R8I,
	InternalUnignedIntRed8       = GL_R8UI,
	InternalIntRed16             = GL_R16I,
	InternalUnignedIntRed16      = GL_R16UI,
	InternalIntRed32             = GL_R32I,
	InternalUnignedIntRed32      = GL_R32UI,
	InternalIntRedGreen8         = GL_RG8I,
	InternalUnignedIntRedGreen8  = GL_RG8UI,
	InternalIntRedGreen16        = GL_RG16I,
	InternalUnignedIntRedGreen16 = GL_RG16UI,
	InternalIntRedGreen32        = GL_RG32I,
	InternalUnignedIntRedGreen32 = GL_RG32UI,
	InternalIntRGB8              = GL_RGB8I,
	InternalUnsignedIntRGB8      = GL_RGB8UI,
	InternalIntRGB16             = GL_RGB16I,
	InternalUnsignedIntRGB16     = GL_RGB16UI,
	InternalIntRGB32             = GL_RGB32I,
	InternalUnsignedIntRGB32     = GL_RGB32UI,
	InternalIntRGBA8             = GL_RGBA8I,
	InternalUnsignedIntRGBA8     = GL_RGBA8UI,
	InternalIntRGBA16            = GL_RGBA16I,
	InternalUnsignedIntRGBA16    = GL_RGBA16UI,
	InternalIntRGBA32            = GL_RGBA32I,
	InternalUnsignedIntRGBA32    = GL_RGBA32UI,
	InternalCompressedRed        = GL_COMPRESSED_RED,
	InternalCompressedRedGreen   = GL_COMPRESSED_RG,
	InternalCompressedRGB        = GL_COMPRESSED_RGB,
	InternalCompressedRGBA       = GL_COMPRESSED_RGBA,
	InternalCompressedSRGB       = GL_COMPRESSED_SRGB,
	InternalCompressedSRGBA      = GL_COMPRESSED_SRGB_ALPHA,
	InternalRGTCRed              = GL_COMPRESSED_RED_RGTC1,
	InternalSignedRGTCRed        = GL_COMPRESSED_SIGNED_RED_RGTC1,
	InternalRGTCRedGreen         = GL_COMPRESSED_RG_RGTC2,
	InternalSignedRGTCRedGreen   = GL_COMPRESSED_SIGNED_RG_RGTC2,
	InternalUnsignedBPTCRGBA     = GL_COMPRESSED_RGBA_BPTC_UNORM,
	InternalUnsignedBPTCSRGBA    = GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM,
	InternalFloatBPTCRGB         = GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT,
	InternalUnsignedFloatBPTCRGB = GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT,
};

enum TextureDataInput
{
	DataUnsignedByte          = GL_UNSIGNED_BYTE,
	DataByte                  = GL_BYTE,
	DataUnsignedShort         = GL_UNSIGNED_SHORT,
	DataShort                 = GL_SHORT,
	DataUnsignedInt           = GL_UNSIGNED_INT,
	DataInt                   = GL_INT,
	DataFloat                 = GL_FLOAT,
	DataUnsignedByte332       = GL_UNSIGNED_BYTE_3_3_2, 
	DataUnsignedByte233Rev    = GL_UNSIGNED_BYTE_2_3_3_REV, 
	DataUnsignedShort565      = GL_UNSIGNED_SHORT_5_6_5, 
	DataUnsignedShort565Rev   = GL_UNSIGNED_SHORT_5_6_5_REV, 
	DataUnsignedShort4444     = GL_UNSIGNED_SHORT_4_4_4_4, 
	DataUnsignedShort4444Rev  = GL_UNSIGNED_SHORT_4_4_4_4_REV, 
	DataUnsignedShort5551     = GL_UNSIGNED_SHORT_5_5_5_1, 
	DataUnsignedShort5551Rev  = GL_UNSIGNED_SHORT_1_5_5_5_REV, 
	DataUnsignedInt8888       = GL_UNSIGNED_INT_8_8_8_8, 
	DataUnsignedInt8888Rev    = GL_UNSIGNED_INT_8_8_8_8_REV, 
	DataUnsignedInt1010102    = GL_UNSIGNED_INT_10_10_10_2,
	DateUnsignedInt1010102Rev = GL_UNSIGNED_INT_2_10_10_10_REV
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

	inline GLuint GetGLTexture() const;

	void CleanUp();

	inline void SetMagFilter(TextureMagFilter value) const;
	inline void SetMinFilter(TextureMinFilter value) const;
	inline void SetWrapBehaviorS(TextureWrapping value) const;
	inline void SetWrapBehaviorT(TextureWrapping value) const;
	inline void GenerateMipmap() const;
	inline void SetFilters(TextureMinFilter minFilter, TextureMagFilter magFilter, 
		TextureWrapping sWrapping, TextureWrapping tWrapping) const;
	inline void SetAnisotropy(const float value);

	inline void Bind(GLuint slot = 0) const;
	void Load(const std::string& filename);
	void CreateEmpty(std::size_t width, std::size_t height, GLenum type, GLint level = 0);
	template <class T> void Load(const std::vector<T>& data, TextureFormatInternal internal, TextureFormat textureFormat, 
								TextureDataInput dataFormat, std::size_t width, std::size_t height);
	template <class T, std::size_t L> void Load(const std::array<T, L>& data, TextureFormatInternal internal, TextureFormat textureFormat, 
																	TextureDataInput dataFormat, std::size_t width = 0, std::size_t height = 0);
};

inline GLuint Texture2D::GetGLTexture() const
{
	return this->texture;
}

void Texture2D::Bind(GLuint slot) const
{
	glActiveTexture(GL_TEXTURE0 + slot);
	glBindTexture(GL_TEXTURE_2D, this->texture);
}

inline void Texture2D::GenerateMipmap() const
{
	glGenerateMipmap(GL_TEXTURE_2D);
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
	glBindTexture(GL_TEXTURE_2D, this->texture);
	this->SetMinFilter(minFilter);
	this->SetMagFilter(magFilter);
	this->SetWrapBehaviorS(sWrapping);
	this->SetWrapBehaviorT(tWrapping);
}

inline void Texture2D::SetAnisotropy(const float value)
{
	float max;
	glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY, &max);
	glTexParameterf(GL_TEXTURE_2D, GL_ARB_texture_filter_anisotropic, (value > max) ? max : value);
}

template<class T> inline void Texture2D::Load(const std::vector<T>& data, TextureFormatInternal internal, TextureFormat textureFormat,
	TextureDataInput dataFormat, std::size_t width, std::size_t height)
{
	this->CleanUp();
	glGenTextures(1, &this->texture);
	if (this->texture)
	{
		glBindTexture(GL_TEXTURE_2D, this->texture);
		glTexImage2D(GL_TEXTURE_2D, 0, (GLenum) internal, (GLsizei) width, (GLsizei) height, 0, (GLenum) textureFormat, (GLenum) dataFormat, data.data());
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	}
}

template<class T, std::size_t L> inline void Texture2D::Load(const std::array<T, L>& data, TextureFormatInternal internal, TextureFormat textureFormat,
																		TextureDataInput dataFormat, std::size_t width, std::size_t height)
{
	this->CleanUp();
	glGenTextures(1, &this->texture);
	if (!width && !height)
	{
		width = height = (std::size_t) sqrt(L);
		if (width * height != L)
		{
			// TODO: LOGO ERROR
		}
	}
	if (this->texture)
	{
		glBindTexture(GL_TEXTURE_2D, this->texture);
		glTexImage2D(GL_TEXTURE_2D, 0, (GLenum) internal, (GLsizei)width, (GLsizei)height, 0, (GLenum)textureFormat, (GLenum)dataFormat, data.data());
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	}
}


#endif //TEXTURE_H