#include "OBJReader.h"
#include <format>

// Only change is disabling OBJL_CONSOLE_OUTPUT by removing the #define
#pragma warning (push)
#pragma warning( disable : 6001 )
#pragma warning( disable : 4244 )
#include <OBJ_loader.h>
#pragma warning (pop)

#include "Vertex.h" 

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <glm/gtx/vec_swizzle.hpp>

glm::vec3 convert(objl::Vector3 left) { return glm::vec3(left.X, left.Y, left.Z); }
glm::vec2 convert(objl::Vector2 left) { return glm::vec2(left.X, left.Y); }

glm::vec2 convert(const aiVector2D& vector) { return glm::vec2(vector.x, vector.y); }
glm::vec3 convert(const aiVector3D& vector) { return glm::vec3(vector.x, vector.y, vector.z); }
glm::vec4 convert(const aiColor4D& vector) { return glm::vec4(vector.r, vector.g, vector.b, vector.a); }

// TODO: Multiple meshes per file and all that jazz
void OBJReader::ReadOBJ(const std::string& filename, ArrayBuffer& elements, ElementArray& index, const std::size_t& id)
{
	objl::Loader loading;
	loading.LoadFile(filename);
	Log(std::format("Loaded \"{}\": Retrieved: {} Meshes, {} Indices, {} Verticies", filename, 
		loading.LoadedMeshes.size(), loading.LoadedIndices.size(), loading.LoadedVertices.size()));
	if (loading.LoadedMeshes.size() > id)
	{
		objl::Mesh mesh = loading.LoadedMeshes[id];
		elements.BufferData(mesh.Vertices, StaticDraw);
		index.BufferData(mesh.Indices, StaticDraw);
	}
	else
	{
		Log("Unable to load model at index " << id);
	}
}

std::vector<MeshPair> OBJReader::ReadOBJ(const std::string& filename)
{
	std::vector<MeshPair> local{};
	ReadOBJ(filename, local);
	return local;
}

void OBJReader::ReadOBJ(const std::string& filename, std::vector<MeshPair>& input)
{
	objl::Loader loading;
	loading.LoadFile(filename);
	Log(std::format("Loaded \"{}\": Retrieved: {} Meshes, {} Indices, {} Verticies", filename,
		loading.LoadedMeshes.size(), loading.LoadedIndices.size(), loading.LoadedVertices.size()));
	input.clear();
	input.reserve(loading.LoadedMeshes.size());
	for (std::size_t i = 0; i < loading.LoadedMeshes.size(); i++)
	{
		ArrayBuffer elements;
		ElementArray index;
		objl::Mesh mesh = loading.LoadedMeshes[i];
		elements.BufferData(mesh.Vertices, StaticDraw);
		index.BufferData(mesh.Indices, StaticDraw);
		input.emplace_back(std::move(elements), std::move(index));
	}
}

MeshData OBJReader::ReadOBJSimple(const std::string& filename)
{
	objl::Loader loading;
	loading.LoadFile(filename);
	Log(std::format("Loaded \"{}\": Retrieved: {} Meshes, {} Indices, {} Verticies", filename,
		loading.LoadedMeshes.size(), loading.LoadedIndices.size(), loading.LoadedVertices.size()));

	MeshData output;
	GLuint indexElement = 0;

	output.vertex.BufferData(loading.LoadedVertices);
	output.index.BufferData(loading.LoadedIndices);
 	for (std::size_t i = 0; i < loading.LoadedMeshes.size(); i++)
	{
		output.rawIndirect.emplace_back(static_cast<GLuint>(loading.LoadedMeshes[i].Indices.size()), 1,
			static_cast<GLuint>(indexElement), 0, static_cast<GLuint>(0));
		indexElement += static_cast<GLuint>(loading.LoadedMeshes[i].Indices.size());
	}
	output.indirect.BufferData(output.rawIndirect, StaticDraw);
	return output;
}

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

template<typename T> void ProcessVertexed(aiMesh* mesh, std::vector<T>& results)
{
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
		results.push_back(current);
	}
}

MeshData MeshThingy(const std::string& filename)
{
	Assimp::Importer importer;
	const aiScene* scene = importer.ReadFile(filename, aiProcess_Triangulate | aiProcess_ImproveCacheLocality
		| aiProcess_CalcTangentSpace | aiProcess_JoinIdenticalVertices);
	if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
	{
		// Loading failed
		std::cout << "Loading failed\n";
		return {};
	}
	MeshData result;
	aiNode *root = scene->mRootNode;
	std::vector<MeshVertex> vertex;
	std::vector<unsigned int> index;
	// Since this is "pure" model loading this might be moved to iterating through scene->mMeshes

	GLuint totalIndexOffset = 0;
	for (unsigned int j = 0; j < root->mNumChildren; j++)
	{
		aiNode* root2 = root->mChildren[j];
		if (!root2)
		{
			continue;
		}
		std::cout << root2->mNumChildren << "\n";
		std::cout << root2->mNumMeshes << "\n";
		std::cout << root2->mName.C_Str() << "\n";
		for (unsigned int i = 0; i < root2->mNumMeshes; i++)
		{
			std::cout << "Mesh " << i << std::endl;
			aiMesh* mesh = scene->mMeshes[root2->mMeshes[i]];
			if (!mesh)
				continue;
			//std::cout << std::format("Mesh {}: with {} vertices", i, mesh->mNumVertices) << std::endl;
			DrawIndirect flames{ 0, 0, 0, static_cast<GLint>(vertex.size()), 0};
			ProcessVertexed<MeshVertex>(mesh, vertex);
			// Buffer it

			// 3 vertices per triangle
			//std::vector<unsigned int> index(mesh->mNumFaces * 3);
			unsigned int numIndicies = 0;
			for (unsigned int x = 0; x < mesh->mNumFaces; x++)
			{
				for (unsigned int y = 0; y < mesh->mFaces[x].mNumIndices; y++)
				{
					index.push_back(mesh->mFaces[x].mIndices[y]);
				}
				numIndicies += mesh->mFaces[x].mNumIndices;
			}
			flames.vertexCount = numIndicies;
			flames.firstVertexIndex = totalIndexOffset;
			//flames.vertexOffset = 0;
			totalIndexOffset += numIndicies;
			result.rawIndirect.push_back(flames);
			//std::cout << mesh->mName.C_Str() << ":" << numIndicies << ":" << mesh->mNumVertices << '\n';
			// Buffer it
		}
	}
	result.vertex.BufferData(vertex);
	result.index.BufferData(index);
	result.indirect.BufferData(result.rawIndirect);
	return result;
}
