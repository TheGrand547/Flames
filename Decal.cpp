#include "Decal.h"

// Since this is essentially in 2d a triangle being collinear is a degenerate triangle and should be ignored
static void ClipToPlane(std::vector<Triangle>& in, std::vector<Triangle>& out, const Plane& plane)
{
	for (const Triangle& inTri : in)
	{
		float orient = -2.3f;
		bool isSplitByPlane = inTri.SplitAndOrientation(plane, orient);

		// If the triangle isn't split by the plane, and is wholly behind/collinear it, then ignore the triangle, it doesn't intersect
		if (!isSplitByPlane && orient <= EPSILON)
		{
			continue;
		}
		// Get the split triangles
		for (const auto& local : inTri.Split(plane, true))
		{
			// Load them into the output
			isSplitByPlane = local.SplitAndOrientation(plane, orient);
			if ((!isSplitByPlane && orient >= 0))
			{
				out.push_back(local);
			}
		}
	}
}

std::vector<Triangle> Decal::ClipTriangleToUniform(const Triangle& triangle, const glm::vec3& scale)
{
	// I don't know if this has any benefit, might poke around this more though
	// Possibly moving the +/- z to the front because, in theory, they should be minimal
	const std::array<Plane, 6> Planes = {
		Plane(glm::vec3( 1.f,  0.f,  0.f), glm::vec3(-scale.x,      0.f,      0.f)),
		Plane(glm::vec3(0.f,  1.f,  0.f), glm::vec3(0.f, -scale.y,      0.f)),
		Plane(glm::vec3(0.f,  0.f,  1.f), glm::vec3(0.f,      0.f, -scale.z)),
		Plane(glm::vec3(-1.f,  0.f,  0.f), glm::vec3( scale.x,      0.f,      0.f)),
		//Plane(glm::vec3( 0.f,  1.f,  0.f), glm::vec3(     0.f, -scale.y,      0.f)),
		Plane(glm::vec3( 0.f, -1.f,  0.f), glm::vec3(     0.f,  scale.y,      0.f)),
		//Plane(glm::vec3( 0.f,  0.f,  1.f), glm::vec3(     0.f,      0.f, -scale.z)),
		Plane(glm::vec3( 0.f,  0.f, -1.f), glm::vec3(     0.f,      0.f,  scale.z)),
	};
	std::vector<Triangle> currentSet = { triangle };
	// Possibly go back to this thing later
	//currentSet.reserve(2);
	for (std::size_t i = 0; i < 6; i++)
	{
		std::vector<Triangle> empty{};
		//empty.reserve(currentSet.size()); // Might be pushing it
		// Clip the current set of triangles to the plane
		ClipToPlane(currentSet, empty, Planes[i]);
		if (empty.size() == 0)
		{
			currentSet.clear();
			break;
		}
		// Ensure the newly clipped triangles are set as the next set of triangles to clip
		currentSet = std::move(empty);
		//std::swap(currentSet, empty);
	}
	return currentSet;
}
