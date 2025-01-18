#include "ExhaustManager.h"
#include <algorithm>
#include "util.h"
#include "Interpolation.h"

void ExhaustManager::AddExhaust(const glm::vec3& position, const glm::vec3& velocity, unsigned int lifetime)
{
	this->particles.emplace_back(position, velocity, lifetime, lifetime);
}

void ExhaustManager::FillBuffer(ArrayBuffer& buffer) const noexcept
{
	std::vector<glm::vec4> output;
	output.reserve(this->particles.size());
	for (const auto& element : this->particles)
	{
		output.emplace_back(element.position, Easing::EaseOutCubic(static_cast<float>(element.ticksLeft) / element.lifetime));
	}
	buffer.BufferData(output);
}

void ExhaustManager::Update() noexcept
{
	std::erase_if(this->particles, 
		[](Exhaust& element)
		{
			element.ticksLeft--;
			element.position += element.velocity * Tick::TimeDelta;
			element.velocity *= 0.99f; // Close enough to workable
			// TOOD: Maybe do some math on creation to determine a "slow down" coefficient that works easily for reducing velocity
			return element.ticksLeft == 0;
		}
	);
}
