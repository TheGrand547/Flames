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
	IDType id;

	void GenerateHash() noexcept;

public:
	std::uint8_t health;
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

	static constexpr float GetScale() noexcept
	{
		return 5.f;
	}

	inline Model GetModel() const noexcept
	{
		return Model{ this->transform, glm::vec3(ClockBrain::GetScale()) };
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

	inline void SetPos(glm::vec3 b) { this->transform.position = b; }

	inline std::size_t GetModulatedTick() const noexcept
	{
		return Level::GetCurrentTick() + this->tickOffset;
	}

	inline std::size_t GetHash() const noexcept
	{
		return this->hash;
	}

	inline IDType GetID() const noexcept
	{
		return this->id;
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
