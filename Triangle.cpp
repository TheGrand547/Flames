#include "Triangle.h"
#include "Plane.h"

// TODO: account for being perfectly split down the middle
bool Triangle::SplitByPlane(const Plane& plane) const
{
	float dotA = plane.Facing(this->vertices[0]), dotB = plane.Facing(this->vertices[1]), dotC = plane.Facing(this->vertices[2]);
	float signA = glm::sign(dotA), signB = glm::sign(dotB), signC = glm::sign(dotC);

	return !((signA == signB && signB == signC) || (signA == 0.f || signB == 0.f || signC == 0.f));

}

std::vector<Triangle> Triangle::Split(const Plane& plane) const
{
	std::vector<Triangle> triangles;
	// TODO: Maybe look into doing this with some funky matrix stuff <- What are you talking about
	// dot plane.normal() * this->vertices() a 1x3 * 3x3 matrix to get the dots, then subtract vec3(plane.constant)
	float dotA = plane.Facing(this->vertices[0]), dotB = plane.Facing(this->vertices[1]), dotC = plane.Facing(this->vertices[2]);
	float signA = glm::sign(dotA), signB = glm::sign(dotB), signC = glm::sign(dotC);
	
	/* Half split sketch
	halfSplitAB = signA == -signB && signC == 0.f;
	halfSplitBC = signB == -signC && signA == 0.f;
	halfSplitAC = signA == -signC && signB == 0.f;
	
	*/
	glm::bvec3 zeroes = glm::equal(glm::sign(glm::vec3(dotA, dotB, dotC)), glm::vec3(0));
	int sum = zeroes[0] + zeroes[1] + zeroes[2];

	// If all the points are on one side of the plane or if more than two points are on the plane
	if ((signA == signB && signB == signC) || (sum > 1))
	{
		// Plane doesn't pass through this triangle, or all are collinear
		triangles.push_back(*this);
	}
	else
	{
		// If the result is -1 then one is positive and the other is negative, thus being split
		bool splitAB = (signA * signB) == -1.f, 
			splitBC  = (signB * signC) == -1.f, 
			splitCA  = (signC * signA) == -1.f;
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
			// Only one of splitAB, splitBC, and splitCA is true
			if (splitAB && signC == 0.f)
			{
				firstLines = lineAB.Split(plane);
				triangles.emplace_back(this->vertices[2], this->vertices[0], firstLines[0].B);
				triangles.emplace_back(firstLines[1].B,   firstLines[1].A,   this->vertices[2]);
			}
			else if (splitCA && signB == 0.f)
			{
				firstLines = lineCA.Split(plane);
				triangles.emplace_back(this->vertices[0], this->vertices[1], firstLines[0].B);
				triangles.emplace_back(this->vertices[2], firstLines[1].B,   this->vertices[1]);
			}
			else // splitBC && signA == 0.f
			{
				firstLines = lineBC.Split(plane);
				triangles.emplace_back(this->vertices[0], this->vertices[1], firstLines[0].B);
				triangles.emplace_back(this->vertices[0], firstLines[1].B,   this->vertices[2]);
			}
			return triangles;
		}
		if (firstLines.size() != 2 || secondLines.size() != 2)
		{
			std::cout << dotA << ":" << dotB << ":" << dotC << std::endl;
			triangles.push_back(*this);
			return triangles;
		}
		triangles.emplace_back(firstLines[0].B,  firstLines[1].A, secondLines[1].B);
		triangles.emplace_back(firstLines[0].A,  firstLines[0].B, secondLines[1].B);
		triangles.emplace_back(secondLines[1].A, firstLines[0].A, secondLines[1].B);
	}
	return triangles;
}
