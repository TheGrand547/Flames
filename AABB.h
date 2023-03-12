#pragma once
#ifndef AXIS_ALIGNED_BOUNDING_CUBE_H
#define AXIS_ALIGNED_BOUNDING_CUBE_H
#include <glm/glm.hpp>
#include <vector>

class AABB
{
protected:
	glm::vec3 negativeBound, positiveBound;
public:
	constexpr AABB();
	constexpr AABB(const glm::vec3& negativeBound, const glm::vec3& positiveBound);
	constexpr AABB(const AABB& other);

	~AABB() = default;

	inline constexpr bool PointInside(const glm::vec3& point) const;
	inline constexpr bool Overlap(const AABB& other) const;
	static constexpr AABB GetAABC(const glm::vec3& left, const glm::vec3& right);
	static constexpr AABB GetAABC(const std::vector<glm::vec3>& points);
};

typedef AABB AxisAlignedBox;

#endif // AXIS_ALIGNED_BOUNDING_CUBE_H