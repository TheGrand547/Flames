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
	static std::vector<Triangle> ClipTrianglesToUniform(const std::vector<Triangle>& triangles, const glm::vec3& scale);
	static std::vector<Triangle> ClipTriangleToUniform(const Triangle& triangle, const glm::vec3& scale);
};

template<> inline Buffer<ArrayBuffer> Decal::GetDecal<OBB>(const OBB& box, const StaticOctTree<OBB>& tree)
{
	glm::vec3 halfs = box.GetScale();
	glm::vec3 center = box.Center();
	// Maybe the other size too
	glm::mat3 inverseViw = glm::mat3(box.GetNormalMatrix());
	glm::mat3 view = glm::transpose(inverseViw);
	std::vector<TextureVertex> transformedResults{};

	for (auto& maybeHit : tree.Search(box.GetAABB()))
	{
		if (maybeHit->Overlap(box))
		{
			for (const Triangle& tri : maybeHit->GetTriangles())
			{
				glm::mat3 local = tri.GetPoints();
				for (glm::length_t i = 0; i < 3; i++)
				{
					local[i] = view * (local[i] - center);
				}
				for (const Triangle& inner : Decal::ClipTriangleToUniform(Triangle(local), halfs))
				{
					glm::mat3 innerLocal = inner.GetPoints();
					glm::vec3 normal = inner.GetNormal();
					for (glm::length_t i = 0; i < 3; i++)
					{
						glm::vec2 older = innerLocal[i] / halfs;
						innerLocal[i] = (inverseViw * innerLocal[i]) + center;
						// Texture coordinates will be (x, y)
						transformedResults.emplace_back<TextureVertex>({ local[i] + normal * 0.001f, older / 2.f + 0.5f });
					}
				}
			}
		}
	}
	Buffer<ArrayBuffer> buffering;
	buffering.BufferData(transformedResults, StaticDraw);
	return buffering;
}


#endif // DECAL_H