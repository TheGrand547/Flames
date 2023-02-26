#include "Wall.h"
#include "glmHelp.h"
#include <iostream>

Wall::Wall(const Model& model) : model(model.GetModelMatrix()), normal(model.GetNormalMatrix()), 
								lowerBound(INFINITY, INFINITY, INFINITY), upperBound(-INFINITY, -INFINITY, -INFINITY), plane(1, 2, 3, 4)
{
	// TOOD: This is dumb don't do this
	glm::vec3 points[] =
	{
		{ 1, 0,  1},
		{ 1, 0, -1},
		{-1, 0,  1},
		{-1, 0, -1}
	};
	std::cout << this->lowerBound << std::endl;
	std::cout << this->upperBound << std::endl;
	for (int i = 0; i < 4; i++)
	{
		glm::vec3 position = this->model * glm::vec4(points[i], 1);
		std::cout << position << std::endl;
		this->lowerBound.x = glm::min(this->lowerBound.x, position.x);
		this->lowerBound.y = glm::min(this->lowerBound.y, position.y);
		this->lowerBound.z = glm::min(this->lowerBound.z, position.z);

		this->upperBound.x = glm::max(this->upperBound.x, position.x);
		this->upperBound.y = glm::max(this->upperBound.y, position.y);
		this->upperBound.z = glm::max(this->upperBound.z, position.z);
	}
	std::cout << this->lowerBound << std::endl;
	std::cout << this->upperBound << std::endl;
	std::cout << this->normal * glm::vec4(0, 1, 0, 0) << std::endl;
	std::cout << glm::vec4(0, 1, 0, 0) * this->normal << std::endl;

	this->plane = Plane(this->normal * glm::vec4(0, 1, 0, 0), this->model * glm::vec4(0, 0, 0, 1));
}
/*
Wall::Wall(const Wall& other) : model(other.model), normal(other.normal), lowerBound(other.lowerBound), 
								upperBound(other.upperBound), plane(other.plane)
{

}
*/
Wall::~Wall()
{

}

bool Wall::Intersection(const glm::vec3& start, const glm::vec3& end) const
{
	if (this->plane.IntersectsNormal(start, end))
	{
		glm::vec3 tester;

		// TODO: not this, this is bad
		glm::vec3 limited[4] =
		{
			{ 1, 0,  1},
			{ 1, 0, -1},
			{-1, 0,  1},
			{-1, 0, -1}
		};
		for (int i = 0; i < 4; i++)
		{
			limited[i] = this->model * glm::vec4(limited[i], 1);
		}

		bool result = false;
		for (int i = 0; i < 2 && !result; i++)
		{
			result = glm::intersectLineTriangle(start, glm::normalize(end - start), limited[i], limited[i + 1], limited[i + 2], tester);
		}

		if (result)
		{
			tester = start + glm::normalize(end - start) * tester;
			return (this->lowerBound.x <= tester.x && tester.x <= this->upperBound.x) &&
				(this->lowerBound.y <= tester.y && tester.y <= this->upperBound.y) &&
				(this->lowerBound.z <= tester.z && tester.z <= this->upperBound.z);
		}
	}
	return false;
}
