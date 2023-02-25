#pragma once
#ifndef WALL_H
#define WALL_H
#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>
#include "Model.h"
#include "Plane.h"

// Static plane based walls
class Wall
{
private:
	glm::mat4 model, normal;
	glm::vec3 lowerBound, upperBound;
	Plane plane;
public:
	Wall(const Model& model);
	Wall(const Wall& other) noexcept = default;
	~Wall();

	bool Intersection(const glm::vec3& start, const glm::vec3& end) const;
};

#endif WALL_H
