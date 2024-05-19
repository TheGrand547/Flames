#include "Decal.h"

// Since this is essentially in 2d a triangle being collinear is a degenerate triangle and should be ignored
static void ClipToPlane(std::vector<Triangle>& in, std::vector<Triangle>& out, const Plane& plane)
{
    for (const Triangle& inTri : in)
    {
        float orient = -2.3f;
        bool isSplitByPlane = inTri.SplitAndOrientation(plane, orient);

        // If the triangle isn't split by the plane, and is wholly behind/collinear it, then ignore the triangle, it doesn't intersect
        if (!isSplitByPlane && orient < 0)
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
                out.push_back(std::move(local));
            }
        }
    }
}

std::vector<Triangle> Decal::ClipTrianglesToUniform(const std::vector<Triangle>& triangles, const glm::vec3& scale)
{
    std::vector<Triangle> splits;
    const std::array<Plane, 6> Planes = { 
        Plane(glm::vec3( 1,  0,  0), glm::vec3(-scale.x,        0,        0)),
        Plane(glm::vec3(-1,  0,  0), glm::vec3( scale.x,        0,        0)),
        Plane(glm::vec3( 0,  1,  0), glm::vec3(       0, -scale.y,        0)),
        Plane(glm::vec3( 0, -1,  0), glm::vec3(       0,  scale.y,        0)),
        Plane(glm::vec3( 0,  0,  1), glm::vec3(       0,        0, -scale.z)),
        Plane(glm::vec3( 0,  0, -1), glm::vec3(       0,        0,  scale.z)),
    };
    for (const Triangle& input: triangles)
    {
        std::vector<Triangle> currentSet = {input};
        for (std::size_t i = 0; i < 6; i++)
        {
            std::vector<Triangle> empty{};
            // Clip the current set of triangles to the plane
            ClipToPlane(currentSet, empty, Planes[i]);
            if (currentSet.size() == 0)
            {
                break;
            }
            // Ensure the newly clipped triangles are set as the next set of triangles to clip
            std::swap(empty, currentSet);
        }
        std::move(currentSet.begin(), currentSet.end(), std::back_inserter(splits));
    }

    return splits;
}

std::vector<Triangle> Decal::ClipTriangleToUniform(const Triangle& triangle, const glm::vec3& scale)
{
    const std::array<Plane, 6> Planes = {
        Plane(glm::vec3( 1,  0,  0), glm::vec3(-scale.x,        0,        0)),
        Plane(glm::vec3(-1,  0,  0), glm::vec3( scale.x,        0,        0)),
        Plane(glm::vec3( 0,  1,  0), glm::vec3(       0, -scale.y,        0)),
        Plane(glm::vec3( 0, -1,  0), glm::vec3(       0,  scale.y,        0)),
        Plane(glm::vec3( 0,  0,  1), glm::vec3(       0,        0, -scale.z)),
        Plane(glm::vec3( 0,  0, -1), glm::vec3(       0,        0,  scale.z)),
    };
    std::vector<Triangle> currentSet = { triangle };
    for (std::size_t i = 0; i < 6; i++)
    {
        std::vector<Triangle> empty{};
        // Clip the current set of triangles to the plane
        ClipToPlane(currentSet, empty, Planes[i]);
        if (empty.size() == 0)
        {
            currentSet.clear();
            break;
        }
        // Ensure the newly clipped triangles are set as the next set of triangles to clip
        currentSet = empty;
    }
    return currentSet;
}
