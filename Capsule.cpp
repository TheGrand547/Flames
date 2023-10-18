#include "Capsule.h"
#include "log.h"

bool Capsule::Intersect(const Capsule& other) const noexcept
{
	Collision temp{};
	return this->Intersect(other, temp);
}

// See: https://wickedengine.net/2020/04/26/capsule-collision-detection/

bool Capsule::Intersect(const Capsule& other, Collision& hit) const noexcept
{
	glm::vec3 bestA(0), bestB(0);
	float distance = this->line.Distance(other.line, bestA, bestB);
	hit.normal = glm::normalize(bestB - bestA);
	hit.distance = this->radius + other.radius - distance; // How far into this capsule the other is
	hit.point = bestB + other.radius * hit.normal; // The point of other furthest into the 
	return distance > 0;
}

bool Capsule::Intersect(const Sphere& other) const noexcept
{
	Collision temp{};
	return this->Intersect(other, temp);
}

bool Capsule::Intersect(const Sphere& other, Collision& hit) const noexcept
{
	glm::vec3 closest = this->line.PointClosestTo(other.center);
	hit.normal = glm::normalize(closest - other.center);
	hit.distance = this->radius + other.radius - glm::length(closest - other.center);
	hit.point = closest + hit.normal * other.radius; 
	return hit.distance > 0;
}

glm::vec3 Capsule::ClosestPoint(const glm::vec3& other) const
{
	return this->line.PointClosestTo(other);
}


void Capsule::GenerateMesh(Buffer<ArrayBuffer>& verts, Buffer<ElementArray>& indicies, float radius, float distance,
	const std::size_t latitudeSlices, const std::size_t longitudeSlices)
{
	if (latitudeSlices == 0 || longitudeSlices == 0 || latitudeSlices >= 500 || longitudeSlices >= 500)
	{
		LogF("Invalid Latitude(%zu) or Longitude(%zu) slice count\n", latitudeSlices, longitudeSlices);
		return;
	}
	std::vector<MeshVertex> points;
	std::vector<GLuint> index;

	// Avoid unnecessary reallocations
	points.reserve((latitudeSlices + 1) * (longitudeSlices + 1));
	index.reserve(6 * longitudeSlices * latitudeSlices);

	const float latitudeStep = glm::two_pi<float>() / (float)latitudeSlices;
	const float longitudeStep = glm::pi<float>() / (float)longitudeSlices;
	
	const float inverse = 1.f / radius;

	for (unsigned int i = 0; i <= longitudeSlices; i++)
	{
		float angle = glm::half_pi<float>() - i * longitudeStep;
		float width = radius * cos(angle);
		float height = radius * sin(angle);
		for (unsigned int j = 0; j <= latitudeSlices; j++)
		{
			float miniAngle = j * latitudeStep;
			glm::vec3 vertex{};
			vertex.x = width * cos(miniAngle) + ((j >= latitudeSlices / 4 && j <= latitudeSlices * 3.f / 4) ? (-distance / 2.f) : (distance / 2.f));
			vertex.y = height;
			vertex.z = width * sin(miniAngle);
			glm::vec2 uvs = { 0.f, 0.f };// { (float)j / latitudeSlices, (float)i / longitudeSlices };
			points.push_back({ vertex, {cos(angle) * cos(miniAngle), sin(angle), cos(angle) * sin(miniAngle)}, uvs });
		}
	}
	for (GLuint i = 0; i < longitudeSlices; i++)
	{
		GLuint first = i * (latitudeSlices + 1);
		GLuint last = first + (latitudeSlices + 1);
		for (GLuint j = 0; j < latitudeSlices; j++, first++, last++)
		{
				index.push_back(first + 1);
				index.push_back(last);
				index.push_back(first);
				index.push_back(last + 1);
				index.push_back(last);
				index.push_back(first + 1);
		}
	}
	verts.Generate();
	verts.BufferData(points, StaticDraw);

	indicies.Generate();
	indicies.BufferData(index, StaticDraw);
}