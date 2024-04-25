#pragma once
#ifndef DECAL_H
#define DECAL_H
#include "Buffer.h"
#include "glmHelp.h"
#include "OrientedBoundingBox.h"
#include "StaticOctTree.h"
#include "Triangle.h"

class Decal
{
public:
	template<class T> static Buffer<ArrayBuffer> GetDecal(const OBB& box, const StaticOctTree<T>& tree);

	// Clips triangles in the 2d plane to the range [-1, 1] x [-1, 1]
	static std::vector<Triangle> ClipTrianglesToUniform(const std::vector<Triangle>& triangles);
};

template<> inline Buffer<ArrayBuffer> Decal::GetDecal<OBB>(const OBB& box, const StaticOctTree<OBB>& tree)
{
	glm::vec3 halfs = box.GetScale();
	// Maybe the other size too

	glm::mat4 projection = glm::ortho(-halfs.x, halfs.x, -halfs.y, halfs.y, 0.1f, 1.f);
	glm::mat4 view = glm::transpose(glm::mat3(box.GetNormalMatrix()));
	view[3] = glm::vec4(-box.Center(), 1.f);
	view = glm::inverse(box.GetNormalMatrix());
	projection = glm::mat4(1.f);

	std::vector<Triangle> tris;
	glm::mat4 projectionView = view;

	for (auto& maybeHit : tree.Search(box.GetAABB()))
	{
		if (maybeHit->Overlap(box))
		{
			for (const Triangle& tri : maybeHit->GetTriangles())
			{
				glm::mat3 local = tri.GetPoints();
				for (glm::length_t i = 0; i < 3; i++)
				{
					glm::vec4 temp = projectionView * glm::vec4(local[i], 1.f);
					local[i] = glm::vec3(temp);
					//std::cout << local[i] << std::endl;
				}
				tris.emplace_back(local);
			}
		}
	}
	glm::mat4 invProjectionView = box.GetNormalMatrix(); // Wrong but trying it anyway

	std::vector<Triangle> results = Decal::ClipTrianglesToUniform(tris);
	std::vector<glm::vec3> transformedResults{};
	for (const Triangle& tri : results)
	{
		glm::mat3 local = tri.GetPoints();
		for (glm::length_t i = 0; i < 3; i++)
		{
			local[i] = invProjectionView * glm::vec4(local[i], 1.f);
			transformedResults.push_back(local[i]);
		}
		//transformedResults.emplace_back(local);
	}
	Buffer<ArrayBuffer> buffering;
	buffering.BufferData(transformedResults, StaticDraw);
	return buffering;
}


#endif // DECAL_H