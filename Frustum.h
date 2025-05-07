#pragma once
#ifndef FRUSTUM_H
#define FRUSTUM_H

#include "glmHelp.h"
#include "Plane.h"
#include "Sphere.h"

struct Frustum
{
private:
	Plane near, far, left, right, top, bottom;
public:
	Frustum(const glm::vec3& cameraPosition, const glm::quat& orientation, const glm::vec2& clippingPlanes) noexcept;
	~Frustum() noexcept = default;

	bool Overlaps(const Sphere& sphere) const noexcept;

};


#endif // FRUSTUM_H
