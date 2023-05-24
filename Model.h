#pragma once
#ifndef MODEL_H
#define MODEL_H
#include <glm/glm.hpp>

struct Model
{
	glm::vec3 rotation, scale, translation;

	constexpr Model(glm::vec3 translation = glm::vec3(0), glm::vec3 rotation = glm::vec3(0), glm::vec3 scale = glm::vec3(1.f, 1.f, 1.f));
	~Model() = default;

	Model(const Model& mode) noexcept = default;

	glm::mat4 GetModelMatrix() const;
	glm::mat4 GetNormalMatrix() const;
};

constexpr Model::Model(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale) : translation(translation), rotation(rotation), scale(scale)
{

}

#endif // MODEL_H
