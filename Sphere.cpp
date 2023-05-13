#include "Sphere.h"
#include <numbers>
#include <vector>
#include "glmHelp.h"

#include <iostream>
std::tuple<GLuint, GLuint, std::size_t>  GenerateSphere(const unsigned int latitudeSlices, const unsigned int longitudeSlices)
{
	struct temp_struct
	{
		glm::vec3 point;
		glm::vec3 norm;
	};
	std::vector<temp_struct> points;
	std::vector<GLuint> index;

	// because they're based on the other one's step
	const float latitudeStep = std::numbers::pi_v<float> * 2.0f / longitudeSlices;
	const float longitudeStep = std::numbers::pi_v<float> / latitudeSlices;

	for (unsigned int i = 0; i <= longitudeSlices; i++)
	{
		float angle = std::numbers::pi_v<float> / 2.f - i * longitudeStep;
		float width = cos(angle);
		float height = sin(angle);
		for (unsigned int j = 0; j <= latitudeSlices; j++)
		{
			float miniAngle = j * latitudeStep;
			glm::vec3 vertex{};
			vertex.x = width * cos(miniAngle);
			vertex.y = height;
			vertex.z = width * sin(miniAngle);
			// TODO: texture and color things
			points.push_back({ vertex, vertex});
			// I don't know why but going off the thing the normal at each vertex *is* the normalized position
			// But since the sphere is radius 1 it all works out
		}
	}
	for (GLuint i = 0; i < latitudeSlices; i++)
	{
		GLuint first = i * (longitudeSlices + 1);
		GLuint last = first + (longitudeSlices + 1);
		for (GLuint j = 0; j < longitudeSlices; j++, first++, last++)
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

	GLuint sphereVerticies = (GLuint) index.size();
	std::cout << "Sphere Indices: " << sphereVerticies << std::endl;
	std::cout << "Sphere Verts: " << points.size() << std::endl;
	GLuint sphereBuffer, sphereIndex;
	glGenBuffers(1, &sphereBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, sphereBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(temp_struct) * points.size(), points.data(), GL_STATIC_DRAW);

	glGenBuffers(1, &sphereIndex);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sphereIndex);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * index.size(), index.data(), GL_STATIC_DRAW);
	return std::make_tuple(sphereBuffer, sphereIndex, index.size());
}