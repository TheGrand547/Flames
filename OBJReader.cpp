#include "OBJReader.h"
#include <format>

#include "Vertex.h" 

/*
void ProcessVertex(aiMesh *mesh)
{
	typedef OverstuffedVertex T;
	constexpr bool HasPosition = std::is_same_v<T, Vertex> || std::is_same_v<T, ColoredVertex> 
		|| std::is_same_v<T, NormalVertex> || std::is_same_v<T, TextureVertex> || 
		std::is_same_v<T, CompleteVertex> || std::is_same_v<T, LargestVertex> || 
		std::is_same_v<T, OverstuffedVertex> || std::is_same_v<T, MeshVertex>;

	constexpr bool HasColor = std::is_same_v<T, ColoredVertex> || std::is_same_v<T, CompleteVertex> || 
		std::is_same_v<T, LargestVertex> || std::is_same_v<T, OverstuffedVertex>;

	constexpr bool HasNormal = std::is_same_v<T, MeshVertex> || std::is_same_v<T, CompleteVertex> ||
		std::is_same_v<T, LargestVertex> || std::is_same_v<T, OverstuffedVertex> || std::is_same_v<T, NormalVertex>;

	constexpr bool HasTangents = std::is_same_v<T, LargestVertex> || std::is_same_v<T, OverstuffedVertex>;
	constexpr bool HasBiTangents = std::is_same_v<T, OverstuffedVertex>;
	constexpr bool HasTexture = std::is_same_v<T, MeshVertex> || std::is_same_v<T, CompleteVertex> ||
		std::is_same_v<T, LargestVertex> || std::is_same_v<T, OverstuffedVertex> || std::is_same_v<T, TextureVertex>;
	std::vector<T> results{ mesh->mNumVertices };
	for (unsigned int i = 0; i < mesh->mNumVertices; i++)
	{
		auto p = mesh->mColors[0];
		auto c = mesh->mVertices[i];
		OverstuffedVertex current{};
		if constexpr (HasPosition)
		{
			if (mesh->HasPositions())
			{
				current.position = convert(mesh->mVertices[i]);
			}
			else
			{
				current.position = glm::vec3(0.f);
			}
		}
		if constexpr (HasNormal)
		{
			if (mesh->HasNormals())
			{
				current.normal = convert(mesh->mNormals[i]);
			}
			else
			{
				current.normal = glm::vec3(0.f, 1.f, 0.f);
			}
		}
		if constexpr (HasColor)
		{
			if (mesh->HasVertexColors(0))
			{
				current.color = glm::xyz(convert(mesh->mColors[0][i]));
			}
			else
			{
				current.color = glm::vec3(1.f);
			}
		}
		if constexpr (HasTexture)
		{
			if (mesh->HasTextureCoords(0))
			{
				current.texture = convert(mesh->mTextureCoords[0][i]);
			}
			else
			{
				current.texture = glm::vec2(0.f, 0.f);
			}
		}
		if constexpr (HasTangents)
		{
			if (mesh->HasTangentsAndBitangents())
			{
				current.tangent = convert(mesh->mTangents[i]);
			}
			else
			{
				current.tangent = glm::vec3(1.f, 0.f, 0.f);
			}
		}
		if constexpr (HasBiTangents)
		{
			if (mesh->HasTangentsAndBitangents())
			{
				current.biTangent = convert(mesh->mBitangents[i]);
			}
			else
			{
				current.biTangent = glm::vec3(0.f, 1.f, 0.f);
			}
		}

	}
}*/

