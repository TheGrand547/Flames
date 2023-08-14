#include "OrientedBoundingBox.h"

OrientedBoundingBox::OrientedBoundingBox(const glm::vec3& euler, const glm::vec3& deltas) : matrix(1.f),
halfs(glm::abs(deltas))
{
	this->Rotate(euler);
}

OrientedBoundingBox::OrientedBoundingBox(const Model& model) : OrientedBoundingBox(model.rotation, model.scale)
{
	this->matrix[3] = glm::vec4(model.translation, 1);
}