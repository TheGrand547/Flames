#pragma once
#ifndef OBJ_READER_H
#define OBJ_READER_H
#include <string>
#include <vector>
#include "Buffer.h"
#include "DrawStruct.h"
#include "VertexArray.h"
#include "Vertex.h" 

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <glm/gtx/vec_swizzle.hpp>

inline glm::vec2 convert(const aiVector2D& vector) { return glm::vec2(vector.x, vector.y); }
inline glm::vec3 convert(const aiVector3D& vector) { return glm::vec3(vector.x, vector.y, vector.z); }
inline glm::vec4 convert(const aiColor4D& vector) { return glm::vec4(vector.r, vector.g, vector.b, vector.a); }

struct MeshPair
{
	ArrayBuffer vertex;
	ElementArray index;
};

struct MeshData
{
	ArrayBuffer vertex;
	ElementArray index;
	Buffer<BufferType::DrawIndirect> indirect;
	std::vector<DrawIndirect> rawIndirect;

	inline void Bind(VAO& vao) const noexcept
	{
		vao.Bind();
		vao.BindArrayBuffer(this->vertex);
		this->index.BindBuffer();
	}
};

struct OBJReader
{
	static void ReadOBJ(const std::string& filename, ArrayBuffer& elements, ElementArray& index, const std::size_t& id = 0);
	static std::vector<MeshPair> ReadOBJ(const std::string& filename);
	static void ReadOBJ(const std::string& filename, std::vector<MeshPair>& input);

	static MeshData ReadOBJSimple(const std::string& filename);

	template<typename T> static void ProcessVertexed(aiMesh* mesh, aiVector3D translation, std::vector<T>& results)
	{
		constexpr bool HasPosition = std::is_same_v<T, Vertex> || std::is_same_v<T, ColoredVertex>
			|| std::is_same_v<T, NormalVertex> || std::is_same_v<T, TextureVertex> ||
			std::is_same_v<T, CompleteVertex> || std::is_same_v<T, LargestVertex> ||
			std::is_same_v<T, OverstuffedVertex> || std::is_same_v<T, MeshVertex> || std::is_same_v<T, NormalMeshVertex>;

		constexpr bool HasColor = std::is_same_v<T, ColoredVertex> || std::is_same_v<T, CompleteVertex> ||
			std::is_same_v<T, LargestVertex> || std::is_same_v<T, OverstuffedVertex>;

		constexpr bool HasNormal = std::is_same_v<T, MeshVertex> || std::is_same_v<T, CompleteVertex> ||
			std::is_same_v<T, LargestVertex> || std::is_same_v<T, OverstuffedVertex> || std::is_same_v<T, NormalVertex> 
			|| std::is_same_v<T, NormalMeshVertex>;

		constexpr bool HasTangents = std::is_same_v<T, LargestVertex> || std::is_same_v<T, OverstuffedVertex> 
			|| std::is_same_v<T, NormalMeshVertex>;
		constexpr bool HasBiTangents = std::is_same_v<T, OverstuffedVertex> || std::is_same_v<T, NormalMeshVertex>;
		constexpr bool HasTexture = std::is_same_v<T, MeshVertex> || std::is_same_v<T, CompleteVertex> ||
			std::is_same_v<T, LargestVertex> || std::is_same_v<T, OverstuffedVertex> || std::is_same_v<T, TextureVertex> 
			|| std::is_same_v<T, NormalMeshVertex>;
		if (!mesh)
		{
			Log("Invalid mesh given.");
			return;
		}

		for (unsigned int i = 0; i < mesh->mNumVertices; i++)
		{
			T current{};
			if constexpr (HasPosition)
			{
				if (mesh->HasPositions())
				{
					current.position = convert(mesh->mVertices[i] + translation);
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
			results.push_back(current);
		}
	}

	template<typename T = MeshVertex>
	static MeshData MeshThingy(const std::string& filename)
	{
		Assimp::Importer importer;
		const aiScene* scene = importer.ReadFile(filename, aiProcess_Triangulate | aiProcess_ImproveCacheLocality
			| aiProcess_CalcTangentSpace | aiProcess_JoinIdenticalVertices);
		if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
		{
			// Loading failed
			Log(std::format("Loading of model file '{}' failed", filename));
			return {};
		}
		MeshData result;
		aiNode* root = scene->mRootNode;
		std::vector<T> vertex;
		std::vector<unsigned int> index;
		std::cout << filename << ":" << scene->mName.C_Str() << ":" << scene->mNumMeshes << 
			":" << root->mNumChildren << ":" << root->mNumMeshes << "\n";

		// Since this is "pure" model loading this might be moved to iterating through scene->mMeshes
		// Sounds like it would work, but it doesn't, unless there are no translations locally. Animations are a huge mess
		ExtractData(scene, root, vertex, index, result.rawIndirect, aiVector3D(0.));
		result.vertex.BufferData(vertex);
		result.index.BufferData(index);
		result.indirect.BufferData(result.rawIndirect);
		return result;
	}

	// TODO: Better version once you get around to data stuff
	template<typename T> static void ExtractData(aiMesh* mesh, std::vector<T>& vertexOut, std::vector<unsigned int>& indexOut, 
		std::vector<DrawIndirect>& indirectOut, aiVector3D translation)
	{
		DrawIndirect flames{ 0, 1, 0, static_cast<GLint>(vertexOut.size()), 0 };
		ProcessVertexed<T>(mesh, translation, vertexOut);
		flames.firstVertexIndex = static_cast<GLuint>(indexOut.size());
		unsigned int numIndicies = 0;
		for (unsigned int x = 0; x < mesh->mNumFaces; x++)
		{
			for (unsigned int y = 0; y < mesh->mFaces[x].mNumIndices; y++)
			{
				indexOut.push_back(mesh->mFaces[x].mIndices[y]);
			}
			numIndicies += mesh->mFaces[x].mNumIndices;
		}
		flames.vertexCount = numIndicies;
		indirectOut.push_back(flames);
	}

	template<typename T> static void ExtractData(const aiScene *scene, aiNode* node, std::vector<T>& vertexOut, std::vector<unsigned int>& indexOut,
		std::vector<DrawIndirect>& indirectOut, aiVector3D translation)
	{
		for (unsigned int i = 0; i < node->mNumMeshes; i++)
		{
			ExtractData(scene->mMeshes[node->mMeshes[i]], vertexOut, indexOut, indirectOut, translation);
		}
		for (unsigned int i = 0; i < node->mNumChildren; i++)
		{
			aiVector3D r, t, s;
			aiNode *child = node->mChildren[i];
			child->mTransformation.Decompose(s, r, t);
			ExtractData(scene, node->mChildren[i], vertexOut, indexOut, indirectOut, translation);
		}
	}

	template<typename T> MeshData ExtractMesh(aiMesh* mesh)
	{
		MeshData result{};
		std::vector<T> vertex;
		std::vector<unsigned int> index;
		ExtractData(mesh, vertex, index, result.rawIndirect, aiVector3D(0.f));
		result.vertex.BufferData(vertex);
		result.index.BufferData(index);
		result.indirect.BufferData(result.rawIndirect);

		return result;
	}
};

#endif // OBJ_READER_H