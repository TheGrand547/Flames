#pragma once
#ifndef CUBE_MAP_H
#define CUBE_MAP_H
#include <array>
#include <string>
#include "Texture.h"

class CubeMap
{
protected:
	GLuint texture;
public:
	CubeMap();
	~CubeMap();

	inline void BindTexture(GLuint slot = 0);

	void CleanUp();
	void Generate(const std::array<std::string, 6> files);
};

inline void CubeMap::BindTexture(GLuint slot)
{
	glBindTextureUnit(slot, this->texture);
}

#endif // CUBE_MAP_H
