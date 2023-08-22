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

bool OrientedBoundingBox::Overlap(const Capsule& other) const
{
	Collision collision{};
	return this->Overlap(other, collision);
}

bool OrientedBoundingBox::Overlap(const Capsule& other, Collision& collide) const
{
	return this->Overlap(Sphere(other.GetRadius(), other.ClosestPoint(this->Center())), collide);
}

// TODO: Standarize what the collision.point thingies mean
bool OrientedBoundingBox::Overlap(const Sphere& other, Collision& collision) const
{
	//glm::vec3 axis = this->Center() - other.center;
	//float distance = glm::length(axis);

	//float projected = 0.f;

	//for (glm::length_t i = 0; i < 3; i++)
	//	projected += glm::abs(glm::dot(glm::vec3(this->matrix[i] * this->halfs[i]), axis));
	////std::cout << "D: " << distance << "\tP: " << projected << "\tR: " << other.radius << "\tN: " << glm::normalize(axis) << std::endl;
	//collision.depth = projected - other.radius;
	//collision.normal = glm::normalize(axis);
	//collision.point = glm::vec3(0); // fuck
	//return projected > other.radius;

	AABB local(this->halfs * 2.f);
	glm::vec3 transformed = this->WorldToLocal(other.center - this->Center());
	Sphere temp{ other.radius, transformed };
	bool result = local.Overlap(temp, collision);
	collision.normal = this->matrix * glm::vec4(collision.normal, 0);
	collision.point  = other.center + collision.normal * collision.depth;
	return result;
}

// World is in normalized coordinates so this is trivial
glm::vec3 OrientedBoundingBox::WorldToLocal(const glm::vec3& in) const
{
	return glm::inverse(glm::mat3(this->matrix)) * in;
}