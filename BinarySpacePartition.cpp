#include "BinarySpacePartition.h"

BinarySpacePartition::BinarySpacePartition() : canonical(glm::vec3(1, 0, 0), glm::vec3(0)), front(nullptr), behind(nullptr)
{

}

BinarySpacePartition::~BinarySpacePartition()
{
	this->ClearBSP();
}

void BinarySpacePartition::AddTriangleInternal(const Triangle& polygon, std::vector<Triangle>& front, std::vector<Triangle>& behind)
{
	// None of the triangles that are being called with this function will be split by the plane
	float result = polygon.GetSpatialRelation(this->canonical); // returns -1 if behind, 0 if collinear, 1 if in front
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

void BinarySpacePartition::ClearBSP()
{
	this->collinear.clear();
	this->front.reset();
	this->behind.reset();
	this->canonical = Plane(glm::vec3(1.f, 0.f, 0.f), glm::vec3(0.f));
}

void BinarySpacePartition::GenerateBSP(std::vector<Triangle>& polygons)
{
	if (polygons.size() == 0)
	{
		return;
	}
	this->ClearBSP();
	this->collinear.push_back(polygons[0]);
	this->canonical = polygons[0].GetPlane();
	std::vector<Triangle> front, behind;
	for (std::size_t i = 1; i < polygons.size(); i++)
	{
		Triangle& current = polygons[i];
		bool splitByPlane = current.SplitByPlane(this->canonical);
		if (!splitByPlane)
		{
			this->AddTriangleInternal(current, front, behind);
		}
		else
		{
			std::vector<Triangle> split = current.Split(this->canonical);
			for (Triangle& polygon : split)
			{
				// All of them are guaranteed to not be split by the plane
				this->AddTriangleInternal(polygon, front, behind);
			}
		}
	}
	if (front.size() > 0)
	{
		if ((this->front = std::make_unique<BSP>()))
			this->front->GenerateBSP(front);
	}
	if (behind.size() > 0)
	{
		if ((this->behind = std::make_unique<BSP>())) 
			this->behind->GenerateBSP(behind);
	}
}

bool BinarySpacePartition::TestPoint(const glm::vec3& point) const
{
	float facing = glm::sign(this->canonical.Facing(point));
	if (facing > 0)
	{
		if (this->front) 
			return this->front->TestPoint(point);
		return true;
	}
	else if (facing < 0)
	{
		if (this->behind) 
			return this->behind->TestPoint(point);
		return false;
	}
	else
	{
		for (const Triangle& triangle : this->collinear)
		{
			if (triangle.ContainsPoint(point))
				return false;
		}
		return false;
	}
	return true;
}

void BinarySpacePartition::AddTriangle(const Triangle& triangle)
{
	float result;
	bool isSplit = triangle.SplitAndOrientation(this->canonical, result);
	if (isSplit)
	{
		for (Triangle& tri : triangle.Split(this->canonical))
		{
			this->AddTriangle(tri);
		}
		return;
	}
	if (result < 0)
	{
		if (!this->behind) 
		{
			if ((this->behind = std::make_unique<BSP>()))
			{
				std::vector temp{ triangle };
				this->behind->GenerateBSP(temp);
			}
		}
		else
		{
			this->behind->AddTriangle(triangle);
		}
	}
	else if (result > 0)
	{
		if (!this->front)
		{
			if ((this->front = std::make_unique<BSP>()))
			{
				std::vector temp{ triangle };
				this->front->GenerateBSP(temp);
			}
		}
		else
		{
			this->front->AddTriangle(triangle);
		}
	}
	else
	{
		this->collinear.push_back(triangle);
	}
}

bool BinarySpacePartition::RayCast(const Ray& ray) const
{
	RayCollision near{};
	return this->RayCast(ray, near);
}


bool BinarySpacePartition::RayCast(const Ray& ray, RayCollision& near) const
{
	near.Clear();

	float distance = this->canonical.Facing(ray.initial);
	float direction = glm::sign(this->canonical.FacingNormal(ray.direction));
	if (distance > 0)
	{
		// In front and pointing away/normal to my plane
		if (direction >= 0)
		{
			return (this->front) ? this->front->RayCast(ray, near) : false;
		}
	}
	else if (distance < 0)
	{
		// Behind and pointing away/normal to my plane
		if (direction <= 0)
		{
			return (this->behind) ? this->behind->RayCast(ray, near) : false;
		}
	}
	else
	{
		// Starts somewhere on this plane
		RayCollision local{};
		local.distance = INFINITY;
		for (const Triangle& triangle : this->collinear)
		{
			if (triangle.RayCast(ray, near) && near.distance < local.distance)
			{
				local = near;
			}
		}
		if (!glm::isinf(local.distance))
		{
			near = local;
			return true;
		}
	}
	// Going to have to check both sides
	RayCollision frontNear{};
	RayCollision behindNear{};
	bool frontCollide = (this->front) ? this->front->RayCast(ray, frontNear) : false;
	bool behindCollide = (this->behind) ? this->behind->RayCast(ray, behindNear) : false;
	if (frontCollide && behindCollide)
	{
		// Pick the closer of the two collisions
		near = (frontNear.depth < behindNear.depth) ? frontNear : behindNear;
		return true;
	}
	else if (frontCollide)
	{
		near = frontNear;
		return true;
	}
	else if (behindCollide)
	{
		near = behindNear;
		return true;
	}
	else
	{
		// Doesn't hit either of the sets in front or behind this plane, thus if it hits this at all it must be one of the collinear tris
		for (const Triangle& triangle : this->collinear)
		{
			if (triangle.RayCast(ray, near))
			{
				return true;
			}
		}
	}
	return false;
}
