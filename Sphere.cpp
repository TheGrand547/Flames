#include "Sphere.h"
#include <numbers>
#include <vector>
#include "glmHelp.h"
#include "Vertex.h"

void GenerateSphereNormals(Buffer<ArrayBuffer>& verts, Buffer<ElementArray>& indicies, 
									const unsigned int latitudeSlices, const unsigned int longitudeSlices)
{
	std::vector<NormalVertex> points;
	std::vector<GLuint> index;

	// because they're based on the other one's step
	const float latitudeStep = std::numbers::pi_v<float> * 2.0f / longitudeSlices;
	const float longitudeStep = std::numbers::pi_v<float> / latitudeSlices;

	for (unsigned int i = 0; i <= longitudeSlices; i++)
	{
		float angle = std::numbers::pi_v<float> / 2.f - i * longitudeStep;
		float width = cos(angle);
		float height = sin(angle);
		for (unsigned int j = 0; j <= latitudeSlices; j++)
		{
			float miniAngle = j * latitudeStep;
			glm::vec3 vertex{};
			vertex.x = width * cos(miniAngle);
			vertex.y = height;
			vertex.z = width * sin(miniAngle);
			points.push_back({ vertex, vertex});
		}
	}
	for (GLuint i = 0; i < latitudeSlices; i++)
	{
		GLuint first = i * (longitudeSlices + 1);
		GLuint last = first + (longitudeSlices + 1);
		for (GLuint j = 0; j < longitudeSlices; j++, first++, last++)
		{
			if (i != 0)
			{
				index.push_back(first + 1);
				index.push_back(last);
				index.push_back(first);
			}
			if (i != (longitudeSlices - 1))
			{
				index.push_back(last + 1);
				index.push_back(last);
				index.push_back(first + 1);
			}
			// TODO: Lines for segment thingies you know
			/*
			lineIndices.push_back(first);
			lineIndices.push_back(last);
			if(i != 0) 
			{
				lineIndices.push_back(first);
				lineIndices.push_back(last + 1);
			}			
			*/
		}
	}
	verts.Generate();
	verts.BufferData(points, StaticDraw);
	
	indicies.Generate();
	indicies.BufferData(index, StaticDraw);
}

void GenerateSphereMesh(Buffer<ArrayBuffer>& verts, Buffer<ElementArray>& indicies,
	const unsigned int latitudeSlices, const unsigned int longitudeSlices)
{
	std::vector<MeshVertex> points;
	std::vector<GLuint> index;

	// because they're based on the other one's step
	const float latitudeStep = std::numbers::pi_v<float> *2.0f / longitudeSlices;
	const float longitudeStep = std::numbers::pi_v<float> / latitudeSlices;

	for (unsigned int i = 0; i <= longitudeSlices; i++)
	{
		float angle = std::numbers::pi_v<float> / 2.f - i * longitudeStep;
		float width = cos(angle);
		float height = sin(angle);
		for (unsigned int j = 0; j <= latitudeSlices; j++)
		{
			float miniAngle = j * latitudeStep;
			glm::vec3 vertex{};
			vertex.x = width * cos(miniAngle);
			vertex.y = height;
			vertex.z = width * sin(miniAngle);
			glm::vec2 uvs = { (float)j / latitudeSlices, (float)i / longitudeSlices };
			points.push_back({ vertex, vertex, uvs});
		}
	}
	for (GLuint i = 0; i < latitudeSlices; i++)
	{
		GLuint first = i * (longitudeSlices + 1);
		GLuint last = first + (longitudeSlices + 1);
		for (GLuint j = 0; j < longitudeSlices; j++, first++, last++)
		{
			if (i != 0)
			{
				index.push_back(first + 1);
				index.push_back(last);
				index.push_back(first);
			}
			if (i != (longitudeSlices - 1))
			{
				index.push_back(last + 1);
				index.push_back(last);
				index.push_back(first + 1);
			}
		}
	}
	verts.Generate();
	verts.BufferData(points, StaticDraw);

	indicies.Generate();
	indicies.BufferData(index, StaticDraw);
}

void GenerateSphere(Buffer<ArrayBuffer>& verts, Buffer<ElementArray>& indicies,
	const unsigned int latitudeSlices, const unsigned int longitudeSlices)
{
	std::vector<Vertex> points;
	std::vector<GLuint> index;

	// because they're based on the other one's step
	const float latitudeStep = std::numbers::pi_v<float> *2.0f / longitudeSlices;
	const float longitudeStep = std::numbers::pi_v<float> / latitudeSlices;

	for (unsigned int i = 0; i <= longitudeSlices; i++)
	{
		float angle = std::numbers::pi_v<float> / 2.f - i * longitudeStep;
		float width = cos(angle);
		float height = sin(angle);
		for (unsigned int j = 0; j <= latitudeSlices; j++)
		{
			float miniAngle = j * latitudeStep;
			glm::vec3 vertex{};
			vertex.x = width * cos(miniAngle);
			vertex.y = height;
			vertex.z = width * sin(miniAngle);
			points.push_back(vertex);
		}
	}
	for (GLuint i = 0; i < latitudeSlices; i++)
	{
		GLuint first = i * (longitudeSlices + 1);
		GLuint last = first + (longitudeSlices + 1);
		for (GLuint j = 0; j < longitudeSlices; j++, first++, last++)
		{
			if (i != 0)
			{
				index.push_back(first + 1);
				index.push_back(last);
				index.push_back(first);
			}
			if (i != (longitudeSlices - 1))
			{
				index.push_back(last + 1);
				index.push_back(last);
				index.push_back(first + 1);
			}
		}
	}
	verts.Generate();
	verts.BufferData(points, StaticDraw);

	indicies.Generate();
	indicies.BufferData(index, StaticDraw);
}