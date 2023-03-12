#include "AABC.h"

AABC::AABC() : negativeBound(INFINITY, INFINITY, INFINITY), positiveBound(-INFINITY, -INFINITY, -INFINITY)
{

}

AABC::AABC(const glm::vec3& negativeBound, const glm::vec3& positiveBound) : negativeBound(negativeBound), positiveBound(positiveBound)
{

}

AABC::AABC(const AABC& other)
{
	this->negativeBound = other.negativeBound;
	this->positiveBound = other.positiveBound;
}

AABC::~AABC()
{

}

inline constexpr bool AABC::PointInside(const glm::vec3& point) const
{
	bool xInside = this->negativeBound.x < point.x && point.x < this->positiveBound.x;
	bool yInside = this->negativeBound.y < point.y && point.y < this->positiveBound.y;
	bool zInside = this->negativeBound.z < point.z && point.z < this->positiveBound.z;
	return xInside && yInside && zInside;
}

constexpr AABC AABC::GetAABC(const glm::vec3& left, const glm::vec3& right)
{
	glm::vec3 min{}, max{};
	
	min.x = glm::min(left.x, right.x);
	min.y = glm::min(left.y, right.y);
	min.z = glm::min(left.z, right.z);

	max.x = glm::max(left.x, right.x);
	max.y = glm::max(left.y, right.y);
	max.z = glm::max(left.z, right.z);

	return AABC(min, max);
}
