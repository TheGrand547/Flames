#include "Test.h"
#include "glmHelp.h"
#include "Triangle.h"
#include "Window.h"
#include <glm/common.hpp>
#include <glm/gtx/vec_swizzle.hpp>
#include <iostream>
#include <fstream>
#include <filesystem>

glm::vec4 TransformToView(glm::vec2 in, glm::mat4 mat, glm::mat2 pat, glm::vec2 size)
{
	glm::vec4 output = mat * glm::vec4((in / size) * 2.f - 1.f, 1.f, 1.f);
	glm::vec2 smo(in / size);
	smo = glm::vec2(smo.x, 1.f - smo.y) * 2.f - 1.f;
	for (int i = 2; i < 10000; i++)
	{
		float x = i / float(10000);
		glm::vec2 dept(x, 1.f);
		output = mat * glm::vec4(smo, dept);
		glm::vec2 screm = pat * glm::vec2(dept);
		assert(output.z / output.w == screm.x / screm.y);
	}
	//std::cout << output.z / output.w << ',' << screm.x / screm.y << '\n';
	output /= output.w;
	return output;
}

void TestFunc()
{
	float zNear = 0.1f, zFar = 250.f;
	std::ofstream sleeper("grump.bin", std::ios::binary);

	glm::vec2 position(0.f);
	glm::vec2 size(1000.f, 0.f);
	glm::vec2 ScreenSize(1000.f);
	glm::mat4 inv = glm::inverse(glm::perspective(glm::radians(70.f), 1.f, zNear, zFar));
	glm::mat2 lower(glm::vec2(inv[2][2], inv[2][3]), glm::vec2(inv[3][2], inv[3][3]));
	glm::vec2 points[4] = { position + glm::yy(size), position + glm::xy(size),
					  position + glm::yx(size), position + glm::xx(size) };
	glm::vec3 points2[4];
	for (int i = 0; i < 4; i++)
	{
		points2[i] = glm::xyz(TransformToView(points[i], inv, lower, ScreenSize));
		//std::cout << points2[i] << '\n';
	}
	const glm::vec3 eye = glm::vec3(0, 0, 0);

	//std::cout << Triangle(eye, points2[2], points2[0]).GetNormal() << '\n'; // Left
	//std::cout << Triangle(eye, points2[1], points2[3]).GetNormal() << '\n'; // Right

	//std::cout << Triangle(eye, points2[1], points2[0]).GetNormal() << '\n'; // Top
	//std::cout << Triangle(eye, points2[3], points2[2]).GetNormal() << '\n'; // Bottom

}