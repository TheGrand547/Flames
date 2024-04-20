#pragma once
#ifndef POLYGON_H
#define POLYGON_H
#include "AABB.h"
#include "OrientedBoundingBox.h"
#include "Triangle.h"

enum class PolygonType
{
	Empty, Triangle, OBB, AABB
};

class Polygon
{
protected:
	PolygonType type = PolygonType::Empty;
	union
	{
		Triangle triangle;
		OBB obb;
		AABB aabb;
	};
public:
	Polygon();
	Polygon(const Triangle& triangle);
	Polygon(const OBB& obb);
	Polygon(const AABB& aabb);
	~Polygon();

	inline bool RayCast(const Ray& ray) const;
	inline bool RayCast(const Ray& ray, RayCollision& collide) const;
	bool RayCast(const Ray& ray, RayCollision& near, RayCollision& far) const;
};

inline bool Polygon::RayCast(const Ray& ray) const
{
	RayCollision near{}, far{};
	return this->RayCast(ray, near, far);
}

inline bool Polygon::RayCast(const Ray& ray, RayCollision& collide) const
{
	RayCollision far{};
	return this->RayCast(ray, collide, far);
}

#endif // POLYGON_H