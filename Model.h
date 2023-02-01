#pragma once
#ifndef MODEL_H
#define MODEL_H
#include <glm.hpp>

struct Model
{
	glm::vec3 rotation, scale, translation;

	Model(glm::vec3 translation = glm::vec3(0), glm::vec3 rotation = glm::vec3(0), glm::vec3 scale = glm::vec3(1.f, 1.f, 1.f));
	~Model();
	glm::mat4 GetModelMatrix() const;
	glm::mat4 GetNormalMatrix() const;
};

#endif // MODEL_H
