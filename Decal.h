#pragma once
#ifndef DECAL_H
#define DECAL_H
#include "Buffer.h"
#include "glmHelp.h"
#include "OrientedBoundingBox.h"
#include "StaticOctTree.h"
#include "Triangle.h"
#include "Geometry.h"

class Decal
{
public:
	template<class T> static ArrayBuffer GetDecal(const OBB& box, const StaticOctTree<T>& tree);
	template<class T> static void GetDecal(const OBB& box, const StaticOctTree<T>& tree, std::vector<TextureVertex>& out);

	// Clips triangles in the 2d plane to the range [-1, 1] x [-1, 1]
	static std::vector<Triangle> ClipTriangleToUniform(const Triangle& triangle, const glm::vec3& scale);
};

template<> inline ArrayBuffer Decal::GetDecal<OBB>(const OBB& box, const StaticOctTree<OBB>& tree)
{
	std::vector<TextureVertex> transformedResults{};
	Decal::GetDecal(box, tree, transformedResults);
	ArrayBuffer buffering;
	buffering.BufferData(transformedResults, StaticDraw);
	return buffering;
}

template<class T> inline void Decal::GetDecal(const OBB& box, const StaticOctTree<T>& tree, std::vector<TextureVertex>& out)
{
	glm::vec3 halfs = box.GetScale();
	glm::vec3 center = box.GetCenter();
	// Maybe the other size too
	glm::mat3 inverseView = glm::mat3(box.GetNormalMatrix());
	glm::mat3 view = glm::transpose(inverseView);

	for (const auto& maybeHit : tree.Search(box.GetAABB()))
	{
		if (DetectCollision::Overlap(*maybeHit, box))
		{
			for (const Triangle& tri : maybeHit->GetTriangles())
			{
				if (!box.Overlap(tri.GetAABB()))
					continue;
				glm::mat3 local = tri.GetPoints();
				glm::vec3 normal = tri.GetNormal();
				if (glm::dot(normal, box.Forward()) > 0.5f)
					continue;
				for (glm::length_t i = 0; i < 3; i++)
				{
					local[i] = view * (local[i] - center);
				}
				for (const Triangle& inner : Decal::ClipTriangleToUniform(Triangle(local), halfs))
				{
					glm::mat3 innerLocal = inner.GetPoints();
					for (glm::length_t i = 0; i < 3; i++)
					{
						glm::vec2 older = glm::vec2(innerLocal[i].z, innerLocal[i].y) / glm::vec2(halfs.x, halfs.y);
						innerLocal[i] = (inverseView * innerLocal[i]) + center;
						// Texture coordinates will be (x, y)
						out.emplace_back<TextureVertex>({ innerLocal[i], older / 2.f + 0.5f });
					}
				}
			}
		}
	}
}


#endif // DECAL_H