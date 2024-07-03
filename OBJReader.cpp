#include "OBJReader.h"

#pragma warning (push)
#pragma warning( disable : 6001 )
#pragma warning( disable : 4244 )
#include <OBJ_loader.h>
#pragma warning (pop)


#include "Vertex.h"

void OBJReader::ReadOBJ(const std::string& filename, ArrayBuffer& elements, ElementArray& index)
{
	objl::Loader loading;
	loading.LoadFile(filename);
	std::cout << loading.LoadedMeshes.size() << std::endl;
	std::cout << loading.LoadedIndices.size() << std::endl;
	std::cout << loading.LoadedVertices.size() << std::endl;
	if (loading.LoadedMeshes.size() > 0)
	{
		objl::Mesh mesh = loading.LoadedMeshes[0];
		elements.BufferData(mesh.Vertices, StaticDraw);
		index.BufferData(mesh.Indices, StaticDraw);
	}
}
