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

	inline Model GetModel() const noexcept
	{
		return Model{ this->transform, glm::vec3(2.f) };
	}

	inline OBB GetOBB() const noexcept
	{
		Model temp = ClockBrain::Collision.GetModel().ApplyParent(this->GetModel());
		return OBB(temp);
	}

	inline MeshMatrix GetPair() const noexcept
	{
		return this->GetModel().GetMatrixPair();
	}

	inline Transform GetTransform() const noexcept
	{
		return this->transform;
	}

	inline std::size_t GetModulatedTick() const noexcept
	{
		return Level::GetCurrentTick() + this->tickOffset;
	}

	static inline OBB Collision;
};

#endif // CLOCK_BRAIN_H
