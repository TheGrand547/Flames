#include "BinarySpacePartition.h"

BinarySpacePartition::BinarySpacePartition() : canonical(glm::vec3(1, 0, 0), glm::vec3(0))
{

}

BinarySpacePartition::~BinarySpacePartition()
{
	this->ClearBSP();
}

void BinarySpacePartition::AddPolygonInternal(const Triangle& polygon, std::vector<Triangle>& front, std::vector<Triangle>& behind)
{
	// None of the triangles that are being called with this function 
	int result = polygon.GetRelation(this->canonical); // returns -1 if behind, 0 if collinear, 1 if in front
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
	if (this->behind)
		delete this->behind;
	if (this->front)
		delete this->front;
	this->front = nullptr;
	this->behind = nullptr;
	this->canonical = Plane(glm::vec3(1, 0, 0), glm::vec3(0));
}

void BinarySpacePartition::GenerateBSP(std::vector<Triangle>& polygons)
{
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
			this->AddPolygonInternal(current, front, behind);
		}
		else
		{
			std::vector<Triangle> split = current.Split(this->canonical);
			for (Triangle& polygon : split)
			{
				// All of them are guaranteed to not be split by the plane
				this->AddPolygonInternal(polygon, front, behind);
			}
		}
	}
	if (front.size() > 0)
	{
		if ((this->front = new BSP())) 
			this->front->GenerateBSP(front);
	}
	if (behind.size() > 0)
	{
		if ((this->behind = new BSP())) 
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
	}
	else if (facing < 0)
	{
		if (this->behind) 
			return this->behind->TestPoint(point);
	}
	else
	{
		for (const Triangle& polygon : this->collinear)
		{
			// if (polygon.ContainsPoint(point))
				return true;
		}
	}
	return false;
}

void BinarySpacePartition::AddPolygon(const Triangle& polygon)
{
	// TODO: this but for things that can be split, should just be a little line of code or two
	float result = polygon.Collinear(this->canonical);
	if (result < 0)
	{
		if (!this->behind) 
		{
			this->behind = new BSP();
			if (this->behind)
			{
				std::vector temp{ polygon };
				this->behind->GenerateBSP(temp);
			}
		}
		else
		{
			this->behind->AddPolygon(polygon);
		}
	}
	else if (result > 0)
	{
		if (!this->front)
		{
			this->front = new BSP();
			if (this->front)
			{
				std::vector temp{ polygon };
				this->front->GenerateBSP(temp);
			}
		}
		else
		{
			this->front->AddPolygon(polygon);
		}
	}
	else
	{
		this->collinear.push_back(polygon);
	}
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
			return (this->front) ? this->front->RayCast(ray, near, far) : false;
		}
	}
	else if (distance < 0)
	{
		// Behind and pointing away/normal to my plane
		if (direction <= 0)
		{
			return (this->behind) ? this->behind->RayCast(ray, near, far) : false;
		}
	}
	else
	{
		for (const Triangle& polygon : this->collinear)
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
