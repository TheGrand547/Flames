#include "AABB.h"

constexpr AABB::AABB() : negativeBound(INFINITY, INFINITY, INFINITY), positiveBound(-INFINITY, -INFINITY, -INFINITY)
{

}

constexpr AABB::AABB(const glm::vec3& negativeBound, const glm::vec3& positiveBound) : negativeBound(negativeBound), positiveBound(positiveBound)
{

}

constexpr AABB::AABB(const AABB& other)
{
	this->negativeBound = other.negativeBound;
	this->positiveBound = other.positiveBound;
}

inline constexpr bool AABB::PointInside(const glm::vec3& point) const
{
	bool xInside = this->negativeBound.x < point.x && point.x < this->positiveBound.x;
	bool yInside = this->negativeBound.y < point.y && point.y < this->positiveBound.y;
	bool zInside = this->negativeBound.z < point.z && point.z < this->positiveBound.z;
	return xInside && yInside && zInside;
}

inline constexpr bool AABB::Overlap(const AABB& other) const
{
	bool xInside = this->negativeBound.x < other.negativeBound.x && this->positiveBound.x > other.negativeBound.x;
	bool yInside = this->negativeBound.y < other.negativeBound.y && this->positiveBound.y > other.negativeBound.y;
	bool zInside = this->negativeBound.z < other.negativeBound.z && this->positiveBound.z > other.negativeBound.z;
	return xInside && yInside && zInside;
}

constexpr AABB AABB::GetAABC(const glm::vec3& left, const glm::vec3& right)
{
	glm::vec3 min{}, max{};
	
	min.x = glm::min(left.x, right.x);
	min.y = glm::min(left.y, right.y);
	min.z = glm::min(left.z, right.z);

	max.x = glm::max(left.x, right.x);
	max.y = glm::max(left.y, right.y);
	max.z = glm::max(left.z, right.z);

	return AABB(min, max);
}

constexpr AABB AABB::GetAABC(const std::vector<glm::vec3>& points)
{
	glm::vec3 min(INFINITY, INFINITY, INFINITY), max(-INFINITY, -INFINITY, -INFINITY);
	for (int i = 0; i < points.size(); i++)
	{
		glm::vec3 point = points[i];
		min.x = glm::min(min.x, point.x);
		min.y = glm::min(min.y, point.y);
		min.z = glm::min(min.z, point.z);

		max.x = glm::max(max.x, point.x);
		max.y = glm::max(max.y, point.y);
		max.z = glm::max(max.z, point.z);
	}
	return AABB(min, max);
}
