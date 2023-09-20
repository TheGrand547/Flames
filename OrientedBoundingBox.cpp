#include "OrientedBoundingBox.h"
#include <bit>

OrientedBoundingBox::OrientedBoundingBox(const glm::vec3& euler, const glm::vec3& deltas) : matrix(1.f), halfs(glm::abs(deltas))
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

static const std::array<const glm::vec3, 8> multiples = {
	{
		{-1.f, -1.f, -1.f},
		{-1.f, -1.f,  1.f},
		{-1.f,  1.f, -1.f},
		{-1.f,  1.f,  1.f},
		{ 1.f, -1.f, -1.f},
		{ 1.f, -1.f,  1.f},
		{ 1.f,  1.f, -1.f},
		{ 1.f,  1.f,  1.f},
	}
};

static const std::array<std::pair<int, int>, 12> linePairs = {
	{
		{0, 1}, //  0, 0, 1
		{0, 2}, //  0, 1, 0 
		{0, 4}, //  1, 0, 0 
		{2, 3}, //  0, 0, 1 
		{1, 3}, //  0, 1, 0 
		{1, 5}, //  1, 0, 0
		{4, 5}, //  0, 0, 1 
		{4, 6}, //  0, 1, 0 
		{2, 6}, //  1, 0, 0 
		{6, 7}, //  0, 0, 1 
		{5, 7}, //  0, 1, 0 
		{3, 7}  //  1, 0, 0 
	}
};

std::vector<LineSegment> OrientedBoundingBox::ClosestFacePoints(const glm::vec3& point) const
{
	std::vector<LineSegment> segments;
	std::vector<glm::vec3> points;
	glm::vec3 center = this->Center();
	float distance = glm::length2(center - point);
	for (glm::length_t i = 0; i < 8; i++)
	{
		glm::vec3 current = center;
		for (glm::length_t j = 0; j < 3; j++)
		{
			current += (*this)[j] * this->halfs[j] * multiples[i][j];
		}
		points.push_back(current);
	}
	for (std::pair<int, int> indexPair : linePairs)
	{
		segments.push_back(LineSegment(points[indexPair.first], points[indexPair.second]));
	}
	return segments;
}
