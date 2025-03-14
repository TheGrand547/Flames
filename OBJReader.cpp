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

// TODO: Switch to assimp, https://github.com/assimp/assimp

void GenerateBitangent(unsigned int a, unsigned int b, unsigned int c, std::vector<glm::vec3>& output, const std::vector<objl::Vertex>& input)
{
	glm::vec3 edgeA = convert(input[b].Position) - convert(input[a].Position);
	glm::vec3 edgeB = convert(input[c].Position) - convert(input[a].Position);

	// Bitangents could also be trivially computed by multiplying the inverse of the column matrix created
	// By the UV deltas on the left of the transpose of the column matrix but this might be easier
	glm::vec2 uvA = convert(input[b].TextureCoordinate) - convert(input[a].TextureCoordinate);
	glm::vec2 uvB = convert(input[c].TextureCoordinate) - convert(input[a].TextureCoordinate);
	float scaling = 1.f / (glm::determinant(glm::mat2{ uvA, uvB }));
	glm::vec3 tangent = scaling * (uvB.y * edgeA - uvA.y * edgeB);
	output.push_back(tangent);
}

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
