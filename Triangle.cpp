#include "Triangle.h"
#include "Plane.h"

std::vector<Triangle> Triangle::Split(const Plane& plane) const
{
	std::vector<Triangle> triangles;
	// TODO: Maybe look into doing this with some funky matrix stuff
	float dotA = plane.Facing(this->vertices[0]), dotB = plane.Facing(this->vertices[1]), dotC = plane.Facing(this->vertices[2]);
	float signA = glm::sign(dotA), signB = glm::sign(dotB), signC = glm::sign(dotC);

	if (signA == signB && signB == signC)
	{
		// Plane doesn't pass through this triangle, or all are collinear
		triangles.push_back(*this);
	}
	else
	{
		// This is going to be sloooooooooooow
		bool splitAB = signA != signB, 
			splitBC = signB != signC, 
			splitAC = signA != signC;
		if (splitAB && splitBC)
		{
			auto lienes = LineSegment(this->vertices[0], this->vertices[1]).Split(plane);
			assert(lienes.size() == 2);
			auto lienes2 = LineSegment(this->vertices[1], this->vertices[2]).Split(plane);
			assert(lienes2.size() == 2);
			// lienes[0].B == lienes[1].B
			// lienes2[0].B == lienes2[1].B
			
			// Single small triangle first
			triangles.emplace_back(lienes[0].B, this->vertices[1], lienes2[1].B);

			triangles.emplace_back(this->vertices[0], lienes[0].B, lienes2[1].B);
			triangles.emplace_back(this->vertices[2], this->vertices[0], lienes2[1].B);
		}
		if (splitAB && splitAC)
		{

		}
		if (splitAC && splitBC)
		{

		}
	}
	return triangles;
}
