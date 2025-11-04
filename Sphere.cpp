#include "Sphere.h"
#include <glm/gtx/compatibility.hpp>
#include <numbers>
#include <vector>
#include "glmHelp.h"
#include "log.h"
#include "Vertex.h"

glm::mat4 Sphere::GetModelMatrix() const noexcept
{
	glm::mat4 temp{ this->radius };
	temp[3] = glm::vec4(this->center, 1);
	return temp;
}

glm::mat4 Sphere::GetNormalMatrix() const noexcept
{
	glm::mat4 temp{ 1 };
	temp[3] = glm::vec4(this->center, 1);
	return temp;
}

void Sphere::GenerateNormals(ArrayBuffer& verts, ElementArray& indicies,
									const std::uint8_t latitudeSlices, const std::uint8_t longitudeSlices) noexcept
{
	if (latitudeSlices == 0 || longitudeSlices == 0)
	{
		LogF("Invalid Latitude(%uhh) or Longitude(%uhh) slice count\n", latitudeSlices, longitudeSlices);
		return;
	}
	std::vector<NormalVertex> points;
	std::vector<GLuint> index;

	std::size_t latSlices = latitudeSlices, longSlices = longitudeSlices;

	// Avoid unnecessary reallocations
	points.reserve(latSlices + 1 * (longSlices + 1));
	index.reserve(6 * (longSlices - 1) * latSlices);

	// because they're based on the other one's step
	const float latitudeStep = glm::two_pi<float>() / static_cast<float>(latSlices);
	const float longitudeStep = glm::pi<float>() / static_cast<float>(longSlices);

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
			if (i + 1 != longitudeSlices)
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

void Sphere::GenerateMesh(ArrayBuffer& verts, ElementArray& indicies,
	const std::uint8_t latitudeSlices, const std::uint8_t longitudeSlices) noexcept
{
	if (latitudeSlices == 0 || longitudeSlices == 0)
	{
		LogF("Invalid Latitude(%uhh) or Longitude(%uhh) slice count\n", latitudeSlices, longitudeSlices);
		return;
	}
	std::vector<MeshVertex> points;
	std::vector<GLuint> index;

	std::size_t latSlices = latitudeSlices, longSlices = longitudeSlices;

	// Avoid unnecessary reallocations
	points.reserve(latSlices + 1 * (longSlices + 1));
	index.reserve(6 * (longSlices - 1) * latSlices);

	// because they're based on the other one's step
	const float latitudeStep = glm::two_pi<float>() / static_cast<float>(latSlices);
	const float longitudeStep = glm::pi<float>() / static_cast<float>(longSlices);

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
			// DON'T TRUST ANY OF THESE UVS
			glm::vec2 uvs = { (float)j / latitudeSlices, (float)i / longitudeSlices };
			//glm::vec2 uvs = { vertex.x / (1 - vertex.y), vertex.z / (1 - vertex.y) }; // Recusively bad mapping
			
			glm::vec2 uvs2 = glm::vec2(std::atan2(fool.z, fool.x) / 2.f, std::asin(fool.y))/ glm::pi<float>() 
				+ glm::vec2(0.5f); // Recusively bad mapping
			if (j == latitudeSlices)
			{
				uvs2.x = 1 - uvs2.x;
			}
			points.push_back({ vertex, vertex, uvs2});
		}
	}
	for (GLuint i = 0; i < longitudeSlices; i++)
	{
		GLuint first = i * (latitudeSlices + 1);
		GLuint last = first + (latitudeSlices + 1);
		for (GLuint j = 0; j < latitudeSlices; j++, first++, last++)
		{
			//if (j + 1 != latitudeSlices)
				//continue;
			if (i != 0)
			{
				index.push_back(first + 1);
				index.push_back(last);
				index.push_back(first);
			}
			if (i + 1 != longitudeSlices)
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

void Sphere::Generate(ArrayBuffer& verts, ElementArray& indicies,
	const std::uint8_t latitudeSlices, const std::uint8_t longitudeSlices) noexcept
{
	if (latitudeSlices == 0 || longitudeSlices == 0)
	{
		LogF("Invalid Latitude(%uhh) or Longitude(%uhh) slice count\n", latitudeSlices, longitudeSlices);
		return;
	}
	std::vector<Vertex> points;
	std::vector<GLuint> index;

	std::size_t latSlices = latitudeSlices, longSlices = longitudeSlices;

	// Avoid unnecessary reallocations
	points.reserve(latSlices + 1 * (longSlices + 1));
	index.reserve(6 * (longSlices - 1) * latSlices);

	// because they're based on the other one's step
	const float latitudeStep = glm::two_pi<float>() / static_cast<float>(latSlices);
	const float longitudeStep = glm::pi<float>() / static_cast<float>(longSlices);

	for (unsigned int i = 0; i <= longSlices; i++)
	{
		float angle = glm::half_pi<float>() - i * longitudeStep;
		float width = cos(angle);
		float height = sin(angle);
		for (unsigned int j = 0; j <= latSlices; j++)
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
			if (i + 1 != longitudeSlices)
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

void Sphere::GenerateLines(ElementArray& indicies, const std::uint8_t latitudeSlices, const std::uint8_t longitudeSlices) noexcept
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