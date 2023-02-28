#include "Wall.h"
#include "glmHelp.h"
#include <iostream>

Wall::Wall(const Model& model) noexcept : model(model.GetModelMatrix()), normal(model.GetNormalMatrix()), plane(1, 2, 3, 4)
{
	this->plane = Plane(this->normal * glm::vec4(0, 1, 0, 0), this->model * glm::vec4(0, 0, 0, 1));
}


Wall::~Wall()
{

}

bool Wall::Intersection(const glm::vec3& start, const glm::vec3& end) const
{
	// Probably bad
	if (this->plane.IntersectsNormal(start, end))
	{
		glm::vec3 tester;

		// TODO: maybe pre-calculate these?
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
		return result;
	}
	return false;
}
