#include "Test.h"`
#include "glmHelp.h"
#include "Triangle.h"
#include "Window.h"
#include <glm/common.hpp>
#include <glm/gtx/vec_swizzle.hpp>
#include <iostream>
#include <fstream>
#include <filesystem>

glm::vec4 TransformToView(glm::vec2 in, glm::mat4 mat, glm::vec2 size)
{
	glm::vec4 output = mat * glm::vec4((in / size) * 2.f - 1.f, 1.f, 1.f);
	output /= output.w;
	return output;
}

void TestFunc()
{
	float zNear = 0.1f, zFar = 250.f;
	std::ofstream sleeper("grump.bin", std::ios::binary);

	glm::vec2 position(20 * 32.f);
	glm::vec2 size(32.f, 0.f);
	glm::vec2 ScreenSize(1000.f);
	glm::mat4 inv = glm::inverse(glm::perspective(glm::radians(70.f), 1.f, zNear, zFar));
	glm::vec2 points[4] = { position + glm::yy(size), position + glm::xy(size),
					  position + glm::yx(size), position + glm::xx(size) };
	glm::vec3 points2[4];
	for (int i = 0; i < 4; i++)
	{
		points2[i] = glm::xyz(TransformToView(points[i], inv, ScreenSize));
	}
	const glm::vec3 eye = glm::vec3(0, 0, 0);

	std::cout << Triangle(eye, points2[2], points2[0]).GetNormal() << '\n'; // Left
	std::cout << Triangle(eye, points2[1], points2[3]).GetNormal() << '\n'; // Right

	std::cout << Triangle(eye, points2[0], points2[1]).GetNormal() << '\n'; // Top
	std::cout << Triangle(eye, points2[3], points2[2]).GetNormal() << '\n'; // Bottom

}