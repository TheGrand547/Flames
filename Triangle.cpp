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
	glm::equal(glm::sign(glm::vec3(dotA, dotB, dotC)), glm::vec3(0));
	
	if ((signA == signB && signB == signC) || (signA == 0.f || signB == 0.f || signC == 0.f))
	{
		// Plane doesn't pass through this triangle, or all are collinear
		triangles.push_back(*this);
	}
	else
	{
		// This is going to be sloooooooooooow
		bool splitAB = signA != signB, 
			splitBC = signB != signC, 
			splitCA = signA != signC;
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
				triangles.emplace_back(  firstLines[1].B,   firstLines[2].A, firstLines[0].B);
			}
			else if (splitCA && signB == 0.f)
			{
				firstLines = LineSegment(this->vertices[2], this->vertices[0]).Split(plane);
			}
			else // splitBC && signA == 0.f
			{
				firstLines = LineSegment(this->vertices[1], this->vertices[2]).Split(plane);
			}
			return triangles;
		}
		if (firstLines.size() != 2 || secondLines.size() != 2)
		{
			std::cout << dotA << ":" << dotB << ":" << dotC << std::endl;
			triangles.push_back(*this);
			return triangles;
		}
		triangles.emplace_back( firstLines[0].B, firstLines[1].A, secondLines[1].B);
		triangles.emplace_back( firstLines[0].A, firstLines[0].B, secondLines[1].B);
		triangles.emplace_back(secondLines[1].A, firstLines[0].A, secondLines[1].B);
	}
	return triangles;
}
