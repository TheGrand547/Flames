#pragma once
#ifndef AXIS_ALIGNED_BOUNDING_CUBE_H
#define AXIS_ALIGNED_BOUNDING_CUBE_H
#include <glm/glm.hpp>

typedef AABC AxisAlignedCube;

class AABC
{
protected:
	glm::vec3 negativeBound, positiveBound;
public:
	AABC();
	AABC(const glm::vec3& negativeBound, const glm::vec3& positiveBound);
	AABC(const AABC& other);
	~AABC();

	inline constexpr bool PointInside(const glm::vec3& point) const;
	static constexpr AABC GetAABC(const glm::vec3& left, const glm::vec3& right);
};

#endif // AXIS_ALIGNED_BOUNDING_CUBE_H