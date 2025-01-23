#pragma once
#ifndef EXHAUST_MANAGER
#define EXHAUST_MANAGER
#include <vector>
#include "glmHelp.h"
#include "Buffer.h"

class ExhaustManager
{
protected:
	struct Exhaust
	{
		glm::vec3 position, velocity;
		unsigned int ticksLeft;
		unsigned int lifetime;
		Exhaust& operator=(const Exhaust& other) noexcept 
		{
			this->position = other.position;
			this->velocity = other.velocity;
			this->ticksLeft = other.ticksLeft;
			this->lifetime = other.lifetime;
			return *this;
		}
	};
	std::vector<Exhaust> particles;
	mutable bool dirty = false;

public:
	ExhaustManager() noexcept = default;
	~ExhaustManager() noexcept = default;
	
	void AddExhaust(const glm::vec3& position, const glm::vec3& velocity, unsigned int lifetime = 128);
	void Update() noexcept;
	void FillBuffer(ArrayBuffer& output) const noexcept;
};

#endif // EXHAUST_MANAGER

