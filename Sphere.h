#pragma once
#ifndef SPHERE_H
#define SPHERE_H
#include <glew.h>
#include <glm/glm.hpp>
#include <tuple> 

// TODO: Sphere CLASS YOU DUMMY
class Sphere
{
};

std::tuple<GLuint, GLuint, std::size_t> GenerateSphere(const int latitudeSlices = 18, const int longitudeSlices = 18);

#endif // SPHERE_H

