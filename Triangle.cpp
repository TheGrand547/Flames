#include "Triangle.h"
#include <glm/gtx/intersect.hpp>
#include "Plane.h"

bool Triangle::SplitAndOrientation(const Plane& plane, float& orientation) const
{
	glm::vec3 dots = plane.Facing(this->vertices);
	glm::bvec3 zeroes = glm::isCompNull(dots, EPSILON);
	// if (glm::none(zeroes)) return glm::all(glm::greater(dots, glm::vec3(0))) || glm::none(glm::greater(dots, glm::vec3(0))
	// else return glm::all(glm::greaterEqual(dots, glm::vec3(0))) || glm::none(glm::greaterEqual(dots, glm::vec3(0))
	/*
	+++ => false, +
	++- => true,  ?
	++0 => false, +

	+-+ => true,  ?
	+-- => true,  ?
	+-0 => true,  ?

	+0+ => false, +
	+0- => true,  ?
	+00 => false, +

	-++ => true,  ?
	-+- => true,  ?
	-+0 => true,  ?

	--+ => true,  ?
	--- => false, -
	--0 => false, -

	-0+ => true,  ?
	-0- => false, -
	-00 => false, -

	0++ => false, +
	0+- => true,  ?
	0+0 => false, +

	0-+ => true,  ?
	0-- => false, -
	0-0 => false, -

	00+ => false, +
	00- => false, -
	000 => false, 0
	*/
	glm::vec3 signs = glm::sign(dots);
	//signs *= glm::not_(zeroes);
	bool fool = false;
	auto transfer = glm::greaterThan(signs, glm::vec3(0));
	if (!glm::any(zeroes))
	{
		fool = (glm::all(transfer) || !glm::any(transfer));
	}
	else
	{
		fool = (glm::all(glm::greaterThanEqual(signs, glm::vec3(0))) ||
			!glm::any(transfer));
	}
	float signify = NAN;
	if (fool) // is split by the plane
	{
		signify = signs[0];
		if (zeroes[0])
		{
			signify = (zeroes[1]) ? signs[2] : signs[1];
		}

	}
	orientation = signify;
	return !fool;
}


bool Triangle::SplitByPlane(const Plane& plane) const
{
	glm::vec3 dots = plane.Facing(this->vertices);
	glm::bvec3 zeroes = glm::isCompNull(dots, EPSILON);
	glm::vec3 signs = glm::sign(dots);
	bool fool = false;
	auto transfer = glm::greaterThan(signs, glm::vec3(0));
	if (!glm::any(zeroes))
	{
		fool = (glm::all(transfer) || !glm::any(transfer));
	}
	else
	{
		fool = (glm::all(glm::greaterThanEqual(signs, glm::vec3(0))) ||
			!glm::any(transfer));
	}
	return fool;
}

float Triangle::GetSpatialRelation(const Plane& plane) const
{
	float result{};
	this->SplitAndOrientation(plane, result);
	return result;
}

bool Triangle::Collinear(const Plane& plane) const
{
	return this->GetSpatialRelation(plane) == 0.f;
}

Plane Triangle::GetPlane() const
{
	LineSegment lineAB(this->vertices[0], this->vertices[1]);
	LineSegment lineBC(this->vertices[1], this->vertices[2]);
	glm::vec3 normal = glm::normalize(glm::cross(lineAB.UnitDirection(), lineBC.UnitDirection()));
	return Plane(normal, this->vertices[2]);
}


std::vector<Triangle> Triangle::Split(const Plane& plane, bool cullBack) const
{
	std::vector<Triangle> triangles;

	glm::vec3 dots = plane.Facing(this->vertices);

	glm::vec3 signs = glm::sign(dots);
	glm::bvec3 zeroes = glm::notEqual(dots, glm::vec3(0), EPSILON);
	signs *= (zeroes);

	float meh;

	// If all the points are on one side of the plane or if more than two points are on the plane
	//if ((signs[0] == signs[1] && signs[1] == signs[2]) || (sum > 1))
	if (!this->SplitAndOrientation(plane, meh))
	{
		// Plane doesn't pass through this triangle, or all are collinear
		triangles.push_back(*this);
	}
	else
	{
		// If the result is -1 then one is positive and the other is negative, thus being split
		bool splitAB = (signs[0] * signs[1]) < 0.f,
			splitBC  = (signs[1] * signs[2]) < 0.f,
			splitCA  = (signs[2] * signs[0]) < 0.f;
		std::vector<LineSegment> firstLines;
		std::vector<LineSegment> secondLines;
		LineSegment lineAB(this->vertices[0], this->vertices[1]);
		LineSegment lineBC(this->vertices[1], this->vertices[2]);
		LineSegment lineCA(this->vertices[2], this->vertices[0]);

		if (splitAB && splitBC)
		{
			firstLines  = lineAB.Split(plane);
			secondLines = lineBC.Split(plane);
		}
		else if (splitCA && splitAB)
		{
			firstLines  = lineCA.Split(plane);
			secondLines = lineAB.Split(plane);
		}
		else if (splitBC && splitCA)
		{
			firstLines  = lineBC.Split(plane);
			secondLines = lineCA.Split(plane);
		}
		else
		{
			// TODO: maybe clean this up idk man
			// Only one of splitAB, splitBC, and splitCA is true
			if (splitAB && !zeroes[2])
			{
				firstLines = lineAB.Split(plane);
				triangles.emplace_back(this->vertices[2], this->vertices[0], firstLines[0].B);
				triangles.emplace_back(this->vertices[2], firstLines[1].B,   firstLines[1].A);
			}
			else if (splitCA && !zeroes[1])
			{
				firstLines = lineCA.Split(plane);
				triangles.emplace_back(this->vertices[1], this->vertices[2], firstLines[1].B);
				triangles.emplace_back(this->vertices[1], firstLines[0].B, this->vertices[0]);
			}
			else // splitBC && !zeroes[0]
			{
				firstLines = lineBC.Split(plane);
				triangles.emplace_back(this->vertices[0], this->vertices[1], firstLines[0].B);
				triangles.emplace_back(this->vertices[0], firstLines[1].B,   this->vertices[2]);
			}
			return triangles;
		}
		if (firstLines.size() != 2 || secondLines.size() != 2)
		{
			std::cout << dots << std::endl;
			triangles.push_back(*this);
			return triangles;
		}
		//triangles.emplace_back(firstLines[0].B,  firstLines[1].A, secondLines[1].B);
		//triangles.emplace_back(firstLines[0].A,  firstLines[0].B, secondLines[1].B);
		//triangles.emplace_back(secondLines[1].A, firstLines[0].A, secondLines[1].B);
		//if (!cullBack || plane.Facing(firstLines[0].A) >= 0.f)
		{
			triangles.emplace_back(firstLines[0].A, firstLines[0].B, secondLines[1].B);
			triangles.emplace_back(secondLines[1].A, firstLines[0].A, secondLines[1].B);
		}
		//if (!cullBack || plane.Facing(firstLines[1].A) >= 0.f)
		{
			triangles.emplace_back(firstLines[0].B, firstLines[1].A, secondLines[1].B);
		}
	}
	return triangles;
}

glm::vec3 Triangle::ClosestPoint(glm::vec3 point) const noexcept
{
	glm::vec3 closestPoint = Plane(this->GetNormal(), this->vertices[0]).GetClosestPoint(point);
	if (this->ContainsPoint(closestPoint))
	{
		return closestPoint;
	}
	glm::vec3 current{};
	float currentDistance = INFINITY;
	for (auto i = 0; i < this->vertices.length(); i++)
	{
		glm::vec3 local = LineSegment(this->vertices[i], this->vertices[(i + 1) % 3]).PointClosestTo(point);
		float distance = glm::length(local);
		if (distance < currentDistance)
		{
			currentDistance = distance;
			current = local;
		}
	}
	return current;
}


