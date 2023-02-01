#pragma once
#ifndef TEXTURE_H
#define TEXTURE_H
#include <glew.h>
#include <glm.hpp>
#include <string>

class Texture
{
private:
	GLuint texture;
	GLenum type;
	int width, height, channels;
	unsigned char* data;
public:
	Texture();
	Texture(const std::string& filename, GLenum type = GL_TEXTURE_2D);
	~Texture();

	void Bind(GLuint slot = 0) const;
	void Load(const std::string& filename, GLenum type = GL_TEXTURE_2D);
};

#endif TEXTURE_H