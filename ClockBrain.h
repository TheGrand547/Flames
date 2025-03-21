#pragma once
#ifndef CLOCK_BRAIN_H
#define CLOCK_BRAIN_H
#include "Transform.h"
#include "OBJReader.h"
#include "VertexArray.h"
#include "AABB.h"

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
	inline AABB GetAABB() const noexcept
	{
		// Probably should make this a constant or something idk
		return AABB(this->transform.position - (0.5f), this->transform.position +(0.5f));
	}

};

#endif // CLOCK_BRAIN_H
