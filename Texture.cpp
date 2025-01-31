#include "Texture.h"
#include "log.h"

static std::string textureBasePath = "textures/";

std::string Texture::GetBasePath()
{
    return textureBasePath;
}

void Texture::SetBasePath(const std::string& path)
{
    textureBasePath = path + FILEPATH_SLASH;
}

GLuint Texture::GetColorChannels(const TextureFormatInternal& format)
{
    switch (format)
    {
    case InternalRed:
    case InternalRed8:
    case InternalSignedRed8:
    case InternalRed16:
    case InternalSignedRed16:
    case InternalFloatRed16:
    case InternalFloatRed32:
    case InternalIntRed8:
    case InternalUnsignedIntRed8:
    case InternalIntRed16:
    case InternalUnsignedIntRed16:
    case InternalIntRed32:
    case InternalUnsignedIntRed32:
    case InternalCompressedRed:
    case InternalDepth:
    case InternalDepth16:
    case InternalDepthFloat32:
    case InternalDepth32:
    case InternalStencil:
    case InternalSignedRGTCRed:
    case InternalRGTCRed:
        return 1;
    case InternalRedGreen:
    case InternalRedGreen16:
    case InternalSignedRedGreen16:
    case InternalFloatRedGreen16:
    case InternalFloatRedGreen32:
    case InternalIntRedGreen8:
    case InternalUnsignedIntRedGreen8:
    case InternalIntRedGreen16:
    case InternalUnsignedIntRedGreen16:
    case InternalIntRedGreen32:
    case InternalUnsignedIntRedGreen32:
    case InternalCompressedRedGreen:
    case InternalRGTCRedGreen:
    case InternalSignedRGTCRedGreen:
    case InternalFloatBPTCRGB:
    case InternalUnsignedFloatBPTCRGB:
        return 2;
    case InternalRGB:
    case InternalRGB332:
    case InternalRGB4:
    case InternalRGB5:
    case InternalRGB8:
    case InternalSignedRGB8:
    case InternalRGB10:
    case InternalRGB12:
    case InternalRGB16:
    case InternalSignedRGB16:
    case InternalSRGB8:
    case InternalFloatRGB16:
    case InternalFloatRGB32:
    case InternalFloatR11G11B10:
    case InternalFloatShared5RGB9:
    case InternalIntRGB8:
    case InternalUnsignedIntRGB8:
    case InternalIntRGB16:
    case InternalUnsignedIntRGB16:
    case InternalIntRGB32:
    case InternalUnsignedIntRGB32:
    case InternalCompressedRGB:
    case InternalCompressedSRGB:
    case InternalUnsignedBPTCSRGBA:
        return 3;
    default:
        return 4;
    }
}

GLuint Texture::GetColorChannels(const TextureFormat& format)
{
    switch (format)
    {
    case FormatRed:
    case FormatDepth:
        return 1;
    case FormatRedGreen:
    case FormatDepthStencil:
        return 2;
    case FormatRGB:
    case FormatBGR:
        return 3;
    default:
        return 4;
    }
}

GLenum Texture::GetSizedInteral(const TextureFormatInternal& format)
{
    switch (format)
    {
    case InternalRGBA:
        return InternalRGBA8;
    case InternalRGB:
        return InternalRGB8;
    case InternalRedGreen:
        return InternalRedGreen8;
    case InternalRed:
        return InternalRed8;
    }
    return format;
}
