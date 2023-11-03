#include "Sphere.h"
#include <numbers>
#include <vector>
#include "glmHelp.h"
#include "log.h"
#include "Vertex.h"

void Sphere::GenerateNormals(Buffer<ArrayBuffer>& verts, Buffer<ElementArray>& indicies, 
									const std::uint8_t latitudeSlices, const std::uint8_t longitudeSlices) noexcept
{
	if (latitudeSlices == 0 || longitudeSlices == 0)
	{
		LogF("Invalid Latitude(%uhh) or Longitude(%uhh) slice count\n", latitudeSlices, longitudeSlices);
		return;
	}
	std::vector<NormalVertex> points;
	std::vector<GLuint> index;

	// Avoid unnecessary reallocations
	points.reserve(std::size_t(latitudeSlices + 1) * std::size_t(longitudeSlices + 1));
	index.reserve(6 * std::size_t(longitudeSlices - 1) * latitudeSlices);

	const float latitudeStep = glm::two_pi<float>() / (float)latitudeSlices;
	const float longitudeStep = glm::pi<float>() / (float)longitudeSlices;

	for (unsigned int i = 0; i <= longitudeSlices; i++)
	{
		float angle = glm::half_pi<float>() - i * longitudeStep;
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
	for (GLuint i = 0; i < longitudeSlices; i++)
	{
		GLuint first = i * (latitudeSlices + 1);
		GLuint last = first + (latitudeSlices + 1);
		for (GLuint j = 0; j < latitudeSlices; j++, first++, last++)
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

void Sphere::GenerateMesh(Buffer<ArrayBuffer>& verts, Buffer<ElementArray>& indicies,
	const std::uint8_t latitudeSlices, const std::uint8_t longitudeSlices) noexcept
{
	if (latitudeSlices == 0 || longitudeSlices == 0)
	{
		LogF("Invalid Latitude(%uhh) or Longitude(%uhh) slice count\n", latitudeSlices, longitudeSlices);
		return;
	}
	std::vector<MeshVertex> points;
	std::vector<GLuint> index;

	// Avoid unnecessary reallocations
	points.reserve(std::size_t(latitudeSlices + 1) * std::size_t(longitudeSlices + 1));
	index.reserve(6 * std::size_t(longitudeSlices - 1) * latitudeSlices);

	const float latitudeStep = glm::two_pi<float>() / (float) latitudeSlices;
	const float longitudeStep = glm::pi<float>() / (float) longitudeSlices;

	for (std::uint8_t i = 0; i <= longitudeSlices; i++)
	{
		float angle = glm::half_pi<float>() - i * longitudeStep;
		float width = cos(angle);
		float height = sin(angle);
		//height += (i >= longitudeSlices / 2) ? -0.5 : 0.5;
		for (std::uint8_t j = 0; j <= latitudeSlices; j++)
		{
			float miniAngle = j * latitudeStep;
			glm::vec3 vertex{};
			vertex.x = width * cos(miniAngle);
			vertex.y = height;
			vertex.z = width * sin(miniAngle);
			glm::vec3 fool = -glm::normalize(vertex);
			glm::vec2 uvs = { (float)j / latitudeSlices, (float)i / longitudeSlices };
			//glm::vec2 uvs = { vertex.x / (1 - vertex.y), vertex.z / (1 - vertex.y) }; // Recusively bad mapping
			//glm::vec2 uvs = { 0.5f + glm::atan(fool.z, fool.x) / glm::two_pi<float>(), 0.5f + glm::asin(fool.y) / glm::pi<float>()}; // Recusively bad mapping
			
			points.push_back({ vertex, vertex, uvs});
		}
	}
	for (GLuint i = 0; i < longitudeSlices; i++)
	{
		GLuint first = i * (latitudeSlices + 1);
		GLuint last = first + (latitudeSlices + 1);
		for (GLuint j = 0; j < latitudeSlices; j++, first++, last++)
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

void Sphere::Generate(Buffer<ArrayBuffer>& verts, Buffer<ElementArray>& indicies,
	const std::uint8_t latitudeSlices, const std::uint8_t longitudeSlices) noexcept
{
	if (latitudeSlices == 0 || longitudeSlices == 0)
	{
		LogF("Invalid Latitude(%uhh) or Longitude(%uhh) slice count\n", latitudeSlices, longitudeSlices);
		return;
	}
	std::vector<Vertex> points;
	std::vector<GLuint> index;

	// TODO: Look into this not playing nice with exception handling bullshit
	// Avoid unnecessary reallocations
	points.reserve(std::size_t(latitudeSlices + 1) * std::size_t(longitudeSlices + 1));
	index.reserve(6 * std::size_t(longitudeSlices - 1) * latitudeSlices);

	// because they're based on the other one's step
	const float latitudeStep = glm::two_pi<float>() / (float)latitudeSlices;
	const float longitudeStep = glm::pi<float>() / (float)longitudeSlices;

	for (unsigned int i = 0; i <= longitudeSlices; i++)
	{
		float angle = glm::half_pi<float>() - i * longitudeStep;
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
	for (GLuint i = 0; i < longitudeSlices; i++)
	{
		GLuint first = i * (latitudeSlices + 1);
		GLuint last = first + (latitudeSlices + 1);
		for (GLuint j = 0; j < latitudeSlices; j++, first++, last++)
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

void Sphere::GenerateLines(Buffer<ElementArray>& indicies, const std::uint8_t latitudeSlices, const std::uint8_t longitudeSlices) noexcept
{
	std::vector<unsigned int> index;
	index.reserve(2 * std::size_t(2 * longitudeSlices - 1) * latitudeSlices);
	for (GLuint i = 0; i < longitudeSlices; i++)
	{
		GLuint first = i * (latitudeSlices + 1);
		GLuint last = first + (latitudeSlices + 1);
		for (GLuint j = 0; j < latitudeSlices; j++, first++, last++)
		{
			index.push_back(first);
			index.push_back(last);
			if (i != 0)
			{
				index.push_back(first);
				index.push_back(first + 1);
			}
		}
	}
	indicies.Generate();
	indicies.BufferData(index, StaticDraw);
}