#include "Geometry.h"

namespace Tetrahedron
{
	static const std::array<glm::vec3, 4> Points =
	{
		glm::vec3( 0.5,    0, -glm::sqrt(2.f) / 4.f),
		glm::vec3(-0.5,    0, -glm::sqrt(2.f) / 4.f),
		glm::vec3(   0,  0.5,  glm::sqrt(2.f) / 4.f),
		glm::vec3(   0, -0.5,  glm::sqrt(2.f) / 4.f),
	};

	static const std::array<unsigned char, 12> Lines =
	{
		0, 1,
		0, 2,
		0, 3,
		1, 2,
		1, 3,
		2, 3
	};

	static const std::array<unsigned char, 12> Triangles =
	{
		0, 1, 2,
		0, 2, 3,
		0, 3, 1,
		1, 3, 2

	};

	std::array<glm::vec3, 4> GetPoints() noexcept
	{
		return Points;
	}

	std::array<unsigned char, 12> GetLineIndex() noexcept
	{
		return Lines;
	}

	std::array<unsigned char, 12> GetTriangleIndex() noexcept
	{
		return Triangles;
	}
}

namespace Cube
{
	static const std::array<glm::vec3, 8> Points =
	{
		glm::vec3{-1, -1, -1},   // -x, -y, -z
		glm::vec3{ 1, -1, -1},   // +x, -y, -z
		glm::vec3{ 1,  1, -1},   // +x, +y, -z
		glm::vec3{-1,  1, -1},   // -x, +y, -z
		glm::vec3{-1, -1,  1},   // -x, -y, +z
		glm::vec3{ 1, -1,  1},   // +x, -y, +z
		glm::vec3{ 1,  1,  1},   // +x, +y, +z
		glm::vec3{-1,  1,  1},   // -x, +y, +z
	};

	std::array<unsigned char, 24> Lines =
	{
		0, 1,  1, 2,  2, 3,  3, 0,
		4, 5,  5, 6,  6, 7,  7, 4,
		2, 6,  5, 1,
		3, 7,  4, 0,
	};

	// I'm not really sure what I was on about here, but it appears to worked for my purposes
	// If j = (index) % 6, then j = 0/4 are unique, j = 1/2 are repeated as 3/5 respectively
	static const std::array<unsigned char, 36> Triangles =
	{
		0, 3, 4, // -X Face
		4, 3, 7,
		0, 4, 1, // -Y Face
		1, 4, 5,
		1, 2, 0, // -Z Face
		0, 2, 3,
		6, 2, 5, // +X Face
		5, 2, 1,
		6, 7, 2, // +Y Face
		2, 7, 3,
		7, 6, 4, // +Z Face
		4, 6, 5,
	};

	// TODO: Fix
	std::array<TextureVertex, 36> UVPoints =
		[](auto verts, auto index)
		{
			std::array<TextureVertex, 36> temp{};
			for (int i = 0; i < temp.size(); i++)
			{
				temp[i].coordinates = verts[index[i]];
			}
			return temp;
		} (Points, Triangles);

	std::array<glm::vec3, 8> GetPoints() noexcept
	{
		return Points;
	}

	std::array<TextureVertex, 36> GetUVPoints() noexcept
	{
		return UVPoints;
	}

	std::array<unsigned char, 24> GetLineIndex() noexcept
	{
		return Lines;
	}

	std::array<unsigned char, 36> GetTriangleIndex() noexcept
	{
		return Triangles;
	}
}

namespace Planes
{
	static const std::array<glm::vec3, 4> Points =
	{
		glm::vec3{ 1, 0,  1},
		glm::vec3{ 1, 0, -1},
		glm::vec3{-1, 0,  1},
		glm::vec3{-1, 0, -1}
	};

	static const std::array<TextureVertex, 4> UVPoints =
	{
		TextureVertex{glm::vec3( 1, 0,  1), glm::vec2(1, 1)},
		TextureVertex{glm::vec3( 1, 0, -1), glm::vec2(1, 0)},
		TextureVertex{glm::vec3(-1, 0,  1), glm::vec2(0, 1)},
		TextureVertex{glm::vec3(-1, 0, -1), glm::vec2(0, 0)}
	};

	static const std::array<unsigned char, 5> Line =
	{
		0, 1, 3, 2, 0
	};

	static const std::array<unsigned char, 6> Triangle =
	{
		0, 1, 2, 2, 1, 3
	};

	std::array<glm::vec3, 4> GetPoints() noexcept
	{
		return Points;
	}

	std::array<unsigned char, 5> GetLineIndex() noexcept
	{
		return Line;
	}

	std::array<unsigned char, 6> GetTriangleIndex() noexcept
	{
		return Triangle;
	}

	std::array<TextureVertex, 4> GetUVPoints() noexcept
	{
		return UVPoints;
	}
}


namespace DetectCollision
{
	bool Overlap(Sphere sphere, Triangle triangle) noexcept
	{
		return glm::distance(sphere.center, triangle.ClosestPoint(sphere.center)) < sphere.radius;
	}

	// Once again, based on the incredible paper https://www.geometrictools.com/Documentation/DynamicCollisionDetection.pdf
	bool Overlap(OBB box, Triangle triangle) noexcept
	{
		const glm::mat3 triPoints = triangle.GetPoints();
		const glm::vec3 delta = (triPoints[0] - box.GetCenter());
		const glm::vec3 edgeA = (triPoints[1] - triPoints[0]), edgeB = (triPoints[2] - triPoints[0]),
			edgeC = (triPoints[2] - triPoints[1]);
		const glm::mat3 triEdges(edgeA, edgeB, edgeC);
		const glm::mat3 triDeltas(delta, triPoints[1] - box.GetCenter(), triPoints[2] - box.GetCenter());
		const glm::vec3 normal = glm::normalize(glm::cross(edgeA, edgeB));
		const glm::mat3 boxAxes(box.Forward(), box.Up(), box.Cross());
		// TODO: NaN check normal
		if (glm::any(glm::isnan(normal)))
		{
			Log("Invalid normal");
			return false;
		}

		const glm::vec3 faceDots = normal * boxAxes;
#ifdef _DEBUG
		glm::vec3 flubber{};
		for (glm::mat3::length_type i = 0; i < 3; i++) 
		{
			flubber[i] = glm::dot(boxAxes[i], normal);
			assert(flubber[i] == faceDots[i]);
		}
#endif // _DEBUG
		// dotProducts[i][j] is equivalent to dot(boxAxes[i], triEdges[j])
		const glm::mat3 dotProducts = glm::transpose(triEdges) * boxAxes;
		for (glm::length_t i = 0; i < 3; i++)
		{
			for (glm::length_t j = 0; j < 3; j++)
			{
				assert(dotProducts[i][j] == glm::dot(boxAxes[i], triEdges[j]));
			}
		}

		const glm::vec3 boxSides = box.GetScale();

		// Axes to be tested:
		// Triangle Normal, OBB Normals, OBB Normals cross Triangle Normals
		for (glm::mat3::length_type i = 0; i < 13; i++)
		{
			glm::vec3 triProjections{ 0.f };
			float boxProjection = 0.f;
			if (i == 0) // Triangle normal
			{
				triProjections = glm::vec3(glm::dot(delta, normal));
				boxProjection = glm::dot(glm::abs(normal * boxAxes), boxSides);
			}
			else if (i < 4) // Box face normal
			{
				// 1,2,3
				glm::mat3::length_type face = i - 1; // To access the correct face
				triProjections = glm::vec3(glm::dot(delta, boxAxes[face]));
				triProjections.y += dotProducts[face][0];
				triProjections.z += dotProducts[face][1];
#ifdef _DEBUG
				glm::vec3 triProjections2{ 0.f };
				triProjections2.x = glm::dot(boxAxes[face], triDeltas[0]);
				triProjections2.y = glm::dot(boxAxes[face], triDeltas[1]);
				triProjections2.z = glm::dot(boxAxes[face], triDeltas[2]);
				if (glm::distance(triProjections, triProjections2) > EPSILON)
				{
					Log("Optimization failed");
					triProjections = triProjections2;
				}
#endif // _DEBUG
				boxProjection = boxSides[face];
			}
			else // Box face normal cross edge
			{
				glm::mat3::length_type j = i - 4;
				glm::mat3::length_type face = j / 3;
				glm::mat3::length_type edge = j % 3;
				const glm::vec3 axis = glm::normalize(glm::cross(boxAxes[face], triEdges[edge]));
				triProjections = axis * triDeltas;
				if (edge == 0)
				{
					boxProjection += glm::abs(dotProducts[2][face]) * boxSides[1];
					boxProjection += glm::abs(dotProducts[1][face]) * boxSides[2];
				}
				else if (edge == 1)
				{
					boxProjection += glm::abs(dotProducts[2][face]) * boxSides[0];
					boxProjection += glm::abs(dotProducts[0][face]) * boxSides[2];
				}
				else
				{
					boxProjection += glm::abs(dotProducts[1][face]) * boxSides[0];
					boxProjection += glm::abs(dotProducts[0][face]) * boxSides[1];
				}
			}
			float low = glm::compMin(triProjections);
			float high = glm::compMax(triProjections);
			
			// Triangle covers interval [low,high], box [-BoxProjection,+BoxProjection]
			if (low > boxProjection || high < -boxProjection)
			{
				return false;
			}
		}
		return true;
	}
}