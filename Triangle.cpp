#include "Triangle.h"
#include "Plane.h"

bool Triangle::SplitAndOrientation(const Plane& plane, float& orientation) const
{
	glm::vec3 dots = plane.Facing(this->vertices);
	glm::vec3 signs = glm::vec3(glm::sign(dots[0]), glm::sign(dots[1]), glm::sign(dots[2]));
	glm::bvec3 zeroes = glm::equal(dots, glm::vec3(0), EPSILON);
	signs *= glm::not_(zeroes);

	// Flag is true means all are on the same side, so flag being false means they are on different ones
	bool flag = true;
	float critera = NAN; // Get something better
	for (int i = 0; i < 3; i++)
	{
		if (zeroes[i])
			continue;
		if (glm::isnan(critera))
		{
			critera = signs[i];
		}
		else
		{
			flag &= (critera == signs[i]);
		}
	}
	if (flag)
	{
		orientation = (glm::isnan(critera)) ? 0.f : critera;
	}
	return !flag;
}


// TODO: Un-sloppify
bool Triangle::SplitByPlane(const Plane& plane) const
{
	glm::vec3 dots = plane.Facing(this->vertices);
	glm::vec3 signs = glm::vec3(glm::sign(dots[0]), glm::sign(dots[1]), glm::sign(dots[2]));
	glm::bvec3 zeroes = glm::equal(dots, glm::vec3(0), EPSILON);
	signs *= glm::not_(zeroes);

	bool flag = true;
	float critera = NAN; // Get something better
	for (int i = 0; i < 3; i++)
	{
		if (zeroes[i])
			continue;
		if (glm::isnan(critera))
		{
			critera = signs[i];
		}
		else
		{
			flag &= (critera == signs[i]);
		}
	}
	return !flag;
}

float Triangle::GetRelation(const Plane& plane) const
{
	glm::vec3 dots = plane.Facing(this->vertices);
	glm::vec3 signs = glm::vec3(glm::sign(dots[0]), glm::sign(dots[1]), glm::sign(dots[2]));
	glm::bvec3 zeroes = glm::equal(dots, glm::vec3(0), EPSILON);
	signs *= glm::not_(zeroes);
	int sign = 0;

	bool flag = true;
	float critera = NAN; // Get something better
	for (int i = 0; i < 3; i++)
	{
		if (zeroes[i])
			continue;
		if (glm::isnan(critera))
		{
			critera = signs[i];
		}
		else
		{
			flag &= (critera == signs[i]);
		}
	}
	// Unrolled
	/*
	flag &= (zeroes[0]) ? true : (glm::isnan(critera) ? (critera = signs[0]) != 0.f : (critera == signs[0]));
	flag &= (zeroes[1]) ? true : (glm::isnan(critera) ? (critera = signs[1]) != 0.f : (critera == signs[1]));
	flag &= (zeroes[2]) ? true : (glm::isnan(critera) ? (critera = signs[2]) != 0.f : (critera == signs[2]));
	*/
	return (glm::isnan(critera)) ? 0.f : critera;
}

bool Triangle::Collinear(const Plane& plane) const
{
	return this->GetRelation(plane) == 0.f;
}

Plane Triangle::GetPlane() const
{
	LineSegment lineAB(this->vertices[0], this->vertices[1]);
	LineSegment lineBC(this->vertices[1], this->vertices[2]);
	glm::vec3 normal = glm::normalize(glm::cross(lineAB.UnitDirection(), lineBC.UnitDirection()));
	return Plane(normal, this->vertices[2]);
}


std::vector<Triangle> Triangle::Split(const Plane& plane) const
{
	std::vector<Triangle> triangles;

	glm::vec3 dots = plane.Facing(this->vertices);

	glm::vec3 signs = glm::vec3(glm::sign(dots[0]), glm::sign(dots[1]), glm::sign(dots[2]));
	glm::bvec3 zeroes = glm::equal(dots, glm::vec3(0), EPSILON);
	signs *= glm::not_(zeroes);
	int sum = zeroes[0] + zeroes[1] + zeroes[2];

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
		bool splitAB = (signs[0] * signs[1]) == -1.f,
			splitBC  = (signs[1] * signs[2]) == -1.f,
			splitCA  = (signs[2] * signs[0]) == -1.f;
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
			if (splitAB && zeroes[2])
			{
				firstLines = lineAB.Split(plane);
				triangles.emplace_back(this->vertices[2], this->vertices[0], firstLines[0].B);
				triangles.emplace_back(this->vertices[2], firstLines[1].B,   firstLines[1].A);
			}
			else if (splitCA && zeroes[1])
			{
				firstLines = lineCA.Split(plane);
				triangles.emplace_back(this->vertices[1], this->vertices[2], firstLines[1].B);
				triangles.emplace_back(this->vertices[1], firstLines[0].B, this->vertices[0]);
			}
			else // splitBC && zeroes[0]
			{
				std::cout << splitAB << ":" << splitBC << ":" << splitCA << ":" << zeroes << std::endl;
				std::cout << signs << std::endl;
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
		triangles.emplace_back(firstLines[0].B,  firstLines[1].A, secondLines[1].B);
		triangles.emplace_back(firstLines[0].A,  firstLines[0].B, secondLines[1].B);
		triangles.emplace_back(secondLines[1].A, firstLines[0].A, secondLines[1].B);
	}
	return triangles;
}
