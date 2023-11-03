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
	int width, height, nrChannels;;
	for (std::size_t i = 0; i < 6; i++)
	{
		unsigned char* data = stbi_load(files[i].c_str(), &width, &height, &nrChannels, 0);
		if (data)
		{
			glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + (GLenum) i,
				0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data
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