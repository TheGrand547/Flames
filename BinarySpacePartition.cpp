#include "BinarySpacePartition.h"

BinarySpacePartition::BinarySpacePartition() : canonical(glm::vec3(1, 0, 0), glm::vec3(0))
{

}

BinarySpacePartition::~BinarySpacePartition()
{
}

void BinarySpacePartition::GenerateBSP(std::vector<Polygon>& polygons)
{
	this->collinear.push_back(polygons[0]);
	// this->canonical = polygons[0].GetPlane();
	std::vector<Polygon> front, behind;
	for (std::size_t i = 1; i < polygons.size(); i++)
	{
		Polygon& current = polygons[i];
		bool splitByPlane = false; // plane.splitting(polygon) returns true if the plane splits it
		if (!splitByPlane)
		{
			int result = 1; // plane.collinear(polygon) returns -1 if behind, 0 if collinear, 1 if in front
			if (result < 0)
			{
				behind.push_back(current);
			}
			else if (result > 0)
			{
				front.push_back(current);
			}
			else
			{
				this->collinear.push_back(current);
			}
		}
		else
		{
			std::vector<Polygon> split; // = polygon.split(plane);
			for (Polygon& polygon : split)
			{
				int result = 1; // plane.collinear(polygon) returns -1 if behind, 0 if collinear, 1 if in front
				if (result < 0)
				{
					behind.push_back(polygon);
				}
				else if (result > 0)
				{
					front.push_back(polygon);
				}
				else
				{
					this->collinear.push_back(polygon);
				}
			}
		}
	}
	if (front.size() > 0)
	{
		this->front = new BSP();
		if (this->front) this->front->GenerateBSP(front);
	}
	if (behind.size() > 0)
	{
		this->behind = new BSP();
		if (this->behind) this->behind->GenerateBSP(front);
	}
}

bool BinarySpacePartition::TestPoint(const glm::vec3& point) const
{
	float facing = glm::sign(this->canonical.Facing(point));
	if (facing > 0)
	{
		if (this->front) 
			return this->front->TestPoint(point);
	}
	else if (facing < 0)
	{
		if (this->behind) 
			return this->behind->TestPoint(point);
	}
	else
	{
		for (const Polygon& polygon : this->collinear)
		{
			// if (polygon.ContainsPoint(point))
				return true;
		}
	}
	return false;
}

bool BinarySpacePartition::RayCast(const Ray& ray) const
{
	RayCollision near{}, far{};
	return this->RayCast(ray, near, far);
}

bool BinarySpacePartition::RayCast(const Ray& ray, RayCollision& collide) const
{
	RayCollision far{};
	return this->RayCast(ray, collide, far);
}

bool BinarySpacePartition::RayCast(const Ray& ray, RayCollision& near, RayCollision& far) const
{
	near.Clear();
	far.Clear();

	float distance = this->canonical.Facing(ray.initial);
	float direction = glm::sign(this->canonical.FacingNormal(ray.direction));
	if (distance > 0)
	{
		// In front and pointing away/normal to my plane
		if (direction >= 0)
		{
			if (this->front) 
				return this->front->RayCast(ray, near, far);
			else 
				return false;
		}
	}
	else if (distance < 0)
	{
		// Behind and pointing away/normal to my plane
		if (direction <= 0)
		{
			if (this->front)
				return this->front->RayCast(ray, near, far);
			else
				return false;
		}
	}
	else
	{
		for (const Polygon& polygon : this->collinear)
		{
			//if (polygon.RayCast(ray, near, far))
			return true;
		}
		return false;
	}
	// Going to have to check both sides
	RayCollision frontNear{}, frontFar{};
	RayCollision behindNear{}, behindFar{};
	bool frontCollide = (this->front) ? this->front->RayCast(ray, frontNear, frontFar) : false;
	bool behindCollide = (this->behind) ? this->behind->RayCast(ray, behindNear, behindFar) : false;
	if (frontCollide && behindCollide)
	{
		// Compare them
		return true;
	}
	else if (frontCollide)
	{
		near = frontNear;
		far = frontFar;
		return true;
	}
	else if (behindCollide)
	{
		near = behindNear;
		far = behindFar;
		return true;
	}

	return false;
}
