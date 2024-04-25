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
            auto lo = inTri.GetPoints();
            std::cout << "Dropped" << lo[0] << lo[1] << lo[2] << std::endl;
            continue;
        }
        // Get the split triangles
        auto locals = inTri.Split(plane);
        for (auto& local : locals)
        {
            // Load them into the output
            isSplitByPlane = local.SplitAndOrientation(plane, orient);
            if ((!isSplitByPlane && orient >= 0))// || isSplitByPlane)
            {
                if (isSplitByPlane)
                {
                    auto lo = local.GetPoints();
                   
                }
                out.push_back(local);
            }
            else
            {
                auto lo = local.GetPoints();
               
            }
        }
    }
}

std::vector<Triangle> Decal::ClipTrianglesToUniform(const std::vector<Triangle>& triangles)
{
    std::vector<Triangle> splits;
    const std::array<Plane, 6> Planes = { 
        Plane(glm::vec3( 1,  0,  0), glm::vec3(-1,  0,  0)),
        Plane(glm::vec3(-1,  0,  0), glm::vec3( 1,  0,  0)),
        Plane(glm::vec3( 0,  1,  0), glm::vec3( 0, -1,  0)),
        Plane(glm::vec3( 0, -1,  0), glm::vec3( 0,  1,  0)),
        Plane(glm::vec3( 0,  0,  1), glm::vec3( 0,  0, -1)),
        Plane(glm::vec3( 0,  0, -1), glm::vec3( 0,  0,  1)),
    };
    for (const Triangle& input: triangles)
    {
        std::vector<Triangle> currentSet = {input};
        for (std::size_t i = 0; i < 6; i++)
        {
            std::vector<Triangle> empty{};
            // Clip the current set of triangles to the plane
            ClipToPlane(currentSet, empty, Planes[i]);
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
