#include "OBJReader.h"
#include <format>

// Only change is disabling OBJL_CONSOLE_OUTPUT by removing the #define
#pragma warning (push)
#pragma warning( disable : 6001 )
#pragma warning( disable : 4244 )
#include <OBJ_loader.h>
#pragma warning (pop)

#include "Vertex.h" 

glm::vec3 convert(objl::Vector3 left) { return glm::vec3(left.X, left.Y, left.Z); }
glm::vec2 convert(objl::Vector2 left) { return glm::vec2(left.X, left.Y); }

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

