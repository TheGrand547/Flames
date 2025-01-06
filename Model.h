#pragma once
#ifndef MODEL_H
#define MODEL_H
#include "glmHelp.h"

struct Model
{
	glm::vec3 scale, translation;
	glm::quat rotation;

	// Euler angles are always assumed to be in DEGREES
	inline Model(glm::vec3 translation = glm::vec3(0), glm::vec3 rotation = glm::vec3(0), glm::vec3 scale = glm::vec3(1.f)) noexcept;
	inline Model(glm::vec3 translation, glm::quat rotation, glm::vec3 scale = glm::vec3(1.f)) noexcept;
	~Model() = default;

	Model(const Model& mode) noexcept = default;

	void Translate(const glm::vec3& distance) noexcept;

	glm::mat4 GetModelMatrix() const noexcept;
	glm::mat4 GetNormalMatrix() const noexcept;
};

inline Model::Model(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale) noexcept : translation(translation), 
				rotation(glm::radians(rotation)), scale(scale)
{

}

inline Model::Model(glm::vec3 translation, glm::quat rotation, glm::vec3 scale) noexcept : translation(translation), rotation(rotation), scale(scale)
{

}

inline void Model::Translate(const glm::vec3& distance) noexcept
{
	this->translation += distance;
}

#endif // MODEL_H
