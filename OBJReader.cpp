#include "OBJReader.h"
#include <format>

// Might be worth putting elsewhere but who cares
#pragma warning (push)
#pragma warning( disable : 6001 )
#pragma warning( disable : 4244 )
#include <OBJ_loader.h>
#pragma warning (pop)

#include "Vertex.h"

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
