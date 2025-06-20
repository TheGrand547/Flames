#pragma once
#ifndef CLOCK_BRAIN_H
#define CLOCK_BRAIN_H
#include "Transform.h"
#include "OBJReader.h"
#include "VertexArray.h"
#include "AABB.h"
#include "kdTree.h"

// TODO: Batching and shit
class ClockBrain
{
protected:
	Transform transform;
	glm::vec3 velocity{}, target{};
	glm::i16vec3 home{};
	float wander;
	int state;
	std::uint8_t tickOffset;
public:
	inline ClockBrain() { this->Init(); }
	void Init();
	void Update(const kdTree<Transform>& transforms);
	void Update2(const kdTree<Transform>& transforms);
	glm::vec3 IndirectUpdate(const kdTree<Transform>& transforms) noexcept;
	void Draw(MeshData& data, VAO& vao, Shader& shader) const;
	inline glm::vec3 GetPos() const noexcept
	{
		return this->transform.position;
	}
	inline glm::vec3 GetHome() const noexcept
	{
		return this->home;
	}
	inline AABB GetAABB() const noexcept
	{
		return this->GetOBB().GetAABB();
	}

	inline OBB GetOBB() const noexcept
	{
		OBB tight = ClockBrain::Collision;
		tight.Rotate(this->GetPair().normal);
		tight.Scale(2.f);
		tight.ReCenter(this->transform.position);
		return tight;
	}

	inline MeshMatrix GetPair() const noexcept
	{
		return Model{ this->transform, glm::vec3(2.f)}.GetMatrixPair();
	}

	inline Transform GetTransform() const noexcept
	{
		return this->transform;
	}

	static inline OBB Collision;
};

#endif // CLOCK_BRAIN_H
