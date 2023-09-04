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
};

constexpr Sphere::Sphere(const float& radius, const glm::vec3& center) : center(center), radius(radius)
{

}

// TODO: Buffer
void GenerateSphere(Buffer<ArrayBuffer>& verts, Buffer<ElementArray>& indicies, 
	const unsigned int latitudeSlices = 18, const unsigned int longitudeSlices = 18);



#endif // SPHERE_H

