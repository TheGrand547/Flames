#pragma once
#ifndef MODEL_H
#define MODEL_H
#include "glmHelp.h"
#include "Transform.h"

struct MeshMatrix
{
	glm::mat4 model, normal;

	inline MeshMatrix() noexcept : model(1.f), normal(1.f) {}
	inline MeshMatrix(const glm::mat4& a, const glm::mat4& b) noexcept : model(a), normal(b) {}
};

struct Model
{
	glm::vec3 scale, translation;
	glm::quat rotation;

	// Euler angles are always assumed to be in DEGREES
	inline Model(const Transform& transform, float scale = 1.f) noexcept : scale(scale), translation(transform.position), rotation(transform.rotation) {}
	inline Model(const Transform& transform, const glm::vec3& scale) noexcept : scale(scale), translation(transform.position), rotation(transform.rotation) {}
	inline Model(glm::vec3 translation = glm::vec3(0), glm::vec3 rotation = glm::vec3(0), glm::vec3 scale = glm::vec3(1.f)) noexcept;
	inline Model(glm::vec3 translation, glm::quat rotation, glm::vec3 scale = glm::vec3(1.f)) noexcept;
	inline Model(glm::vec3 translation, glm::quat rotation, float scale) noexcept;
	~Model() = default;

	Model(const Model& mode) noexcept = default;

	void Translate(const glm::vec3& distance) noexcept;

	// Apply a 'parent' transform to the current model, and return it. Translation of *this is interpreted as being in local space
	// Equivalent to multipling the Model Matrices
	Model ApplyParent(const Model& parent) noexcept;

	// Applies the parent rotation to this, then applies the translations linearly. Both translations are are in world space
	Model ApplyParentInGlobal(const Model& parent) noexcept;

	glm::mat4 GetModelMatrix() const noexcept;
	glm::mat4 GetNormalMatrix() const noexcept;
	MeshMatrix GetMatrixPair() const noexcept;
};

inline Model::Model(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale) noexcept : translation(translation), 
				rotation(glm::radians(rotation)), scale(scale)
{

}

inline Model::Model(glm::vec3 translation, glm::quat rotation, glm::vec3 scale) noexcept : translation(translation), rotation(rotation), scale(scale)
{

}

inline Model::Model(glm::vec3 translation, glm::quat rotation, float scale) noexcept : translation(translation), rotation(rotation),
				scale(glm::vec3(scale))
{
}

inline void Model::Translate(const glm::vec3& distance) noexcept
{
	this->translation += distance;
}

#endif // MODEL_H
