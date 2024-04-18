#pragma once
#ifndef BINARY_SPACE_PARITION_H
#define BINARY_SPACE_PARITION_H
#include <vector>
#include "CollisionTypes.h"
#include "glmHelp.h"
#include "Plane.h"
#include "Lines.h"


struct Polygon {}; // placeholder

class BinarySpacePartition
{
using BSP = BinarySpacePartition;

protected:
	Plane canonical;
	std::vector<Polygon> collinear; // Likely to be small
	BSP* front = nullptr, *behind = nullptr;
public:
	BinarySpacePartition();
	~BinarySpacePartition();

	void GenerateBSP(std::vector<Polygon>& polygons);

	// True -> point is contained within the geometry enclosed by this BSP, false -> it is not
	bool TestPoint(const glm::vec3& point) const;

	bool RayCast(const Ray& ray) const;
	bool RayCast(const Ray& ray, RayCollision& collide) const;
	bool RayCast(const Ray& ray, RayCollision& near, RayCollision& far) const;
};

#endif // BINARY_SPACE_PARTITION_H
