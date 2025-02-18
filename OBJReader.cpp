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

MeshData OBJReader::ReadOBJSimple(const std::string& filename)
{
	objl::Loader loading;
	loading.LoadFile(filename);
	Log(std::format("Loaded \"{}\": Retrieved: {} Meshes, {} Indices, {} Verticies", filename,
		loading.LoadedMeshes.size(), loading.LoadedIndices.size(), loading.LoadedVertices.size()));

	MeshData output;
	std::size_t vertexSize = 0, indexSize = 0;

	for (std::size_t i = 0; i < loading.LoadedMeshes.size(); i++)
	{
		vertexSize += loading.LoadedMeshes[i].Vertices.size();
		indexSize += loading.LoadedMeshes[i].Indices.size();
	}
	GLintptr vertexStride = sizeof(loading.LoadedMeshes[0].Vertices.front());
	GLintptr indexStride = sizeof(loading.LoadedMeshes[0].Indices.front());
	output.vertex.Reserve(vertexSize * vertexStride, StaticDraw);
	output.index.Reserve(indexSize * indexStride, StaticDraw);

	GLintptr vertexOffset = 0, indexOffset = 0;
	GLuint vertexElement = 0, indexElement = 0;

	std::vector<Elements> stored;

	for (std::size_t i = 0; i < loading.LoadedMeshes.size(); i++)
	{
		output.vertex.BufferSubData(loading.LoadedMeshes[i].Vertices, vertexOffset);
		output.index.BufferSubData(loading.LoadedMeshes[i].Indices, indexOffset);

		stored.emplace_back(static_cast<GLuint>(loading.LoadedMeshes[i].Vertices.size()), 12,
			static_cast<GLuint>(indexElement), static_cast<GLuint>(vertexElement), static_cast<GLuint>(3));
		auto& ref = stored.back();

		// TODO: outstream overload
		std::cout << ref.vertexCount << ":" << ref.instanceCount << ":" << ref.firstVertexIndex << ":" << ref.vertexOffset << ":" << ref.instanceOffset << "\n";
		vertexOffset += loading.LoadedMeshes[i].Vertices.size() * vertexStride;
		indexOffset  += loading.LoadedMeshes[i].Indices.size() * indexStride;
		vertexElement += loading.LoadedMeshes[i].Vertices.size();
		indexElement += loading.LoadedMeshes[i].Indices.size();
	}
	output.indirect.BufferData(stored, StaticDraw);
	return output;
}
