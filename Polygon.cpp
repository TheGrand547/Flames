#include "Polygon.h"

Polygon::Polygon() : type(PolygonType::Empty), aabb()
{
}

Polygon::Polygon(const Triangle& triangle) : type(PolygonType::Triangle), triangle(triangle)
{
}

Polygon::Polygon(const OBB& obb) : type(PolygonType::OBB), obb(obb)
{
}

Polygon::Polygon(const AABB& aabb) : type(PolygonType::AABB), aabb(aabb)
{
}

Polygon::~Polygon()
{
}

bool Polygon::RayCast(const Ray& ray, RayCollision& near , RayCollision& far) const
{
	switch (this->type)
	{
	//case PolygonType::Triangle: return this->triangle.RayCast(ray, near, far);
		break;
	}
	return false;
}
