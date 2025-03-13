#pragma once
#ifndef CLOCK_BRAIN_H
#define CLOCK_BRAIN_H
#include "Transform.h"
#include "OBJReader.h"
#include "VertexArray.h"

// TODO: Batching and shit
class ClockBrain
{
protected:
	Transform transform;
	glm::vec3 velocity, target;

public:

	void Init();
	void Update();
	void Draw(MeshData& data, VAO& vao, Shader& shader);
	inline glm::vec3 GetPos() const noexcept
	{
		return this->transform.position;
	}

};

#endif // CLOCK_BRAIN_H
