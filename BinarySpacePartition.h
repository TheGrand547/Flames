#pragma once
#ifndef BINARY_SPACE_PARITION_H
#define BINARY_SPACE_PARITION_H
#include <vector>
#include <memory>
#include "CollisionTypes.h"
#include "glmHelp.h"
#include "Lines.h"
#include "Plane.h"
#include "Triangle.h"


class BinarySpacePartition
{
protected:
	Plane canonical;
	std::vector<Triangle> collinear; // Likely to be small
	std::unique_ptr<BinarySpacePartition> front, behind;

	void AddTriangleInternal(const Triangle& triangle, std::vector<Triangle>& front, std::vector<Triangle>& back);
public:
	BinarySpacePartition();
	~BinarySpacePartition();

	void ClearBSP();
	void GenerateBSP(std::vector<Triangle>& triangles);

	// True -> point is "in bounds", False -> point is "out of bounds"
	bool TestPoint(const glm::vec3& point) const;

	// Should be used as little as possible, much less efficicent than doing it via bulk in BuildBSP
	void AddTriangle(const Triangle& triangle);

	bool RayCast(const Ray& ray) const;
	bool RayCast(const Ray& ray, RayCollision& collide) const;
};

using BSP = BinarySpacePartition;


#endif // BINARY_SPACE_PARTITION_H
