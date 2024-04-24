#include "Decal.h"

// Since this is essentially in 2d a triangle being collinear is a degenerate triangle and should be ignored
static void ClipToPlane(std::vector<Triangle>& in, std::vector<Triangle>& out, const Plane& plane)
{
    for (const Triangle& inTri : in)
    {
        // If the triangle isn't split by the plane, and is wholly behind/collinear it, then ignore the triangle, it doesn't intersect
        float orientation = inTri.GetRelation(plane);
        if (!inTri.SplitByPlane(plane) && inTri.GetRelation(plane) < 0.f)
        {
            out.push_back(inTri); // I don't think this'll work
            continue;
        }
        // Get the split triangles
        auto locals = inTri.Split(plane);
        for (auto& local : locals)
        {
            // Load them into the output
            out.push_back(local);
        }
    }
}

std::vector<Triangle> Decal::ClipTrianglesToUniform(const std::vector<Triangle>& triangles)
{
    std::vector<Triangle> splits;
    const std::array<Plane, 4> Planes = { 
        Plane(glm::vec3( 1,  0, 0), glm::vec3(-1,  0, 0)),
        Plane(glm::vec3(-1,  0, 0), glm::vec3( 1,  0, 0)),
        Plane(glm::vec3( 0, -1, 0), glm::vec3( 0,  1, 0)),
        Plane(glm::vec3( 0,  1, 0), glm::vec3( 0, -1, 0)),
    };
    for (const Triangle& input: triangles)
    {
        std::vector<Triangle> currentSet = {input};
        for (std::size_t i = 0; i < 4; i++)
        {
            std::vector<Triangle> empty{};
            //Before(currentSet.size() << ":" << empty.size());
            // Clip the current set of triangles to the plane
            ClipToPlane(currentSet, empty, Planes[i]);
            //After(currentSet.size() << ":" << empty.size());
            // Ensure the newly clipped triangles are set as the next set of triangles to clip
            std::swap(empty, currentSet);
            if (currentSet.size() == 0)
            {
                break;
            }
        }
        std::copy(currentSet.begin(), currentSet.end(), std::back_inserter(splits));
    }

    return splits;
}
