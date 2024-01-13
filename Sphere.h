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

	constexpr void Translate(const glm::vec3& amount) noexcept;
	constexpr void Scale(const float& amount) noexcept;

	glm::mat4 GetModelMatrix() const noexcept;
	glm::mat4 GetNormalMatrix() const noexcept;

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

constexpr void Sphere::Translate(const glm::vec3& amount) noexcept
{
	this->center += radius;
}

constexpr void Sphere::Scale(const float& amount) noexcept
{
	this->radius *= amount;
}

#endif // SPHERE_H

