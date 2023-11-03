#pragma once
#ifndef TEXTURE_H
#define TEXTURE_H
#include <glew.h>

// TODO: Base texture class so I don't have to rewrite things as much smh

enum TextureMagFilter
{
	// "Pixel Perfect"
	MagNearest = GL_NEAREST,
	// Bilinear
	MagLinear  = GL_LINEAR,
};

enum TextureMinFilter
{
	// Quantized
	MinNearest     = GL_NEAREST,
	// Bilinear, no Mipmaping
	MinLinear      = GL_LINEAR,

	// Quantized, with quantized mipmapping
	NearestNearest = GL_NEAREST_MIPMAP_NEAREST,
	// Quantized, with linear mipmapping
	NearestLinear  = GL_NEAREST_MIPMAP_LINEAR,
	// Bilinear, with quantized mipmapping
	LinearNearest  = GL_LINEAR_MIPMAP_NEAREST,
	// Bilinear, with linear mipmapping - aka Trilinear
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
	InternalDepth                = GL_DEPTH_COMPONENT,
	InternalDepth16              = GL_DEPTH_COMPONENT16,
	InternalDepth24              = GL_DEPTH_COMPONENT24,
	InternalDepthFloat32         = GL_DEPTH_COMPONENT32F,
	InternalDepth32              = GL_DEPTH_COMPONENT32,
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

	// FLOAT IS *VERY* SLOW COMPARED TO THE OTHER ONES -- USER BEWARE
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
	InternalUnspecified          = 666,
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
	DataUnsignedShort1555     = GL_UNSIGNED_SHORT_1_5_5_5_REV,
	DataUnsignedInt8888       = GL_UNSIGNED_INT_8_8_8_8,
	DataUnsignedInt8888Rev    = GL_UNSIGNED_INT_8_8_8_8_REV,
	DataUnsignedInt1010102    = GL_UNSIGNED_INT_10_10_10_2,
	DateUnsignedInt1010102Rev = GL_UNSIGNED_INT_2_10_10_10_REV
};

#endif // TEXTURE_H