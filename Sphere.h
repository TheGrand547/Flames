#pragma once
#ifndef SPHERE_H
#define SPHERE_H
#include <glew.h>
#include <glm/glm.hpp>
#include <tuple> 

// TODO: Sphere CLASS YOU DUMMY
class Sphere
{
protected:
	glm::vec3 center;
	float radius;

public:
	constexpr Sphere(const float& radius = 1.f, const glm::vec3& center = glm::vec3(0));
};

constexpr Sphere::Sphere(const float& radius, const glm::vec3& center) : center(center), radius(radius)
{

}

std::tuple<GLuint, GLuint, std::size_t> GenerateSphere(const unsigned int latitudeSlices = 18, const unsigned int longitudeSlices = 18);

#endif // SPHERE_H

