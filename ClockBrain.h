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
	std::size_t hash;

	void GenerateHash() noexcept;

public:
	inline ClockBrain() { this->Init(); }
	// Everything about this is a horrific hack
	void Init(glm::vec3 init = glm::vec3(0.f));
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
		//Model temp = this->GetModel().ApplyParent(ClockBrain::Collision.GetModel());
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

	inline std::size_t GetHash() const noexcept
	{
		return this->hash;
	}

	static inline OBB Collision;
};

template<>
struct std::hash<ClockBrain>
{
	std::size_t operator()(const ClockBrain& brain) const
	{
		return brain.GetHash();
	}
};

#endif // CLOCK_BRAIN_H
