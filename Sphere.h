#pragma once
#ifndef SPHERE_H
#define SPHERE_H
#include <glew.h>
#include <glm/glm.hpp>
#include <tuple>
#include "Buffer.h"
#include "Vertex.h"

// TODO: Sphere CLASS YOU DUMMY
struct Sphere
{
	glm::vec3 center;
	float radius;
	constexpr Sphere(const float& radius = 1.f, const glm::vec3& center = glm::vec3(0));

	static void GenerateNormals(Buffer<ArrayBuffer>& verts, Buffer<ElementArray>& indicies,
		const std::uint8_t latitudeSlices = 18, const std::uint8_t longitudeSlices = 18) noexcept;
	static void GenerateMesh(Buffer<ArrayBuffer>& verts, Buffer<ElementArray>& indicies,
		const std::uint8_t latitudeSlices = 18, const std::uint8_t longitudeSlices = 18) noexcept;
	static void Generate(Buffer<ArrayBuffer>& verts, Buffer<ElementArray>& indicies,
		const std::uint8_t latitudeSlices = 18, const std::uint8_t longitudeSlices = 18) noexcept;
	static void GenerateLines(Buffer<ElementArray>& indicies, const std::uint8_t latitudeSlices = 18, const std::uint8_t longitudeSlices = 18) noexcept;
};

constexpr Sphere::Sphere(const float& radius, const glm::vec3& center) : center(center), radius(radius)
{

}

#endif // SPHERE_H

