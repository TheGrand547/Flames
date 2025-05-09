#include "CubeMap.h"
#include "log.h"
#include "stb_image.h"

CubeMap::CubeMap() : texture(0)
{

}

CubeMap::~CubeMap()
{
	this->CleanUp();
}

void CubeMap::CleanUp()
{
	if (this->texture)
	{
		glDeleteTextures(1, &this->texture);
		this->texture = 0;
	}
}

void CubeMap::Generate(const std::array<Texture2D, 6>& textures)
{
	this->CleanUp();
	glGenTextures(1, &this->texture);
	if (!this->texture)
	{
		Log("Unable to generate cubemap texture");
		return;
	}
	glBindTexture(GL_TEXTURE_CUBE_MAP, this->texture);
	glBindTexture(GL_TEXTURE_CUBE_MAP, this->texture);
	GLsizei width = textures[0].GetWidth(), height = textures[0].GetHeight();
	glTexStorage2D(GL_TEXTURE_CUBE_MAP, 1, 
		Texture::GetSizedInteral(static_cast<TextureFormatInternal>(textures[0].GetFormat())), width, height);
	//glTextureStorage2D(this->texture, 1, InternalRGB8, width, height);
	CheckError();
	for (std::size_t i = 0; i < 6; i++)
	{
		const Texture2D& current = textures[i];

		GLenum targetTexture = GL_TEXTURE_CUBE_MAP_POSITIVE_X + static_cast<GLenum>(i);
		/*
		glTexImage2D(targetTexture, 0, InternalRGB, width, height, BORDER_PARAMETER,
			FormatRGB, GL_UNSIGNED_BYTE, nullptr
		);*/
		glCopyImageSubData(current.GetGLTexture(), GL_TEXTURE_2D, 0, 0, 0, 0,
			this->texture, GL_TEXTURE_CUBE_MAP, 0, 0, 0, static_cast<GLint>(i), 
			width, height, 1);
	}
	// TODO: Maybe allow for different filtering modes? don't think they're needed but eh
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
}

void CubeMap::Generate(const std::array<std::string, 6> files)
{
	this->CleanUp();
	glGenTextures(1, &this->texture);
	if (!this->texture)
	{
		Log("Unable to generate cubemap texture");
		return;
	}
	glBindTexture(GL_TEXTURE_CUBE_MAP, this->texture);
	int width, height, nrChannels;
	std::string path = Texture::GetBasePath();
	for (std::size_t i = 0; i < 6; i++)
	{
		unsigned char* data = stbi_load((path + files[i]).c_str(), &width, &height, &nrChannels, 0);
		if (data)
		{
			glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + static_cast<GLenum>(i),
				0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, static_cast<const void*>(data)
			);
		}
		else
		{
			LogF("Failed to load file '%s' for cubemap\n", files[i].c_str());
		}
		stbi_image_free(data);
	}
	// TODO: Maybe allow for different filtering modes? don't think they're needed but eh
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
}