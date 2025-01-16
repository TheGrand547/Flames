#include "ExhaustManager.h"
#include "util.h"

void ExhaustManager::AddExhaust(const glm::vec3& position, const glm::vec3& velocity, unsigned int lifetime)
{
	this->particles.emplace_back(position, velocity, lifetime, lifetime);
}

void ExhaustManager::Update(ArrayBuffer& output) noexcept
{
	std::vector<glm::vec4> results;
	results.reserve(this->particles.size());
	std::erase_if(this->particles, 
		[&results](Exhaust& element)
		{
			element.ticksLeft--;
			element.position += element.velocity * Tick::TimeDelta;
			// TODO: Easing?
			results.emplace_back(element.position, float(element.ticksLeft) / element.lifetime);
			return element.ticksLeft == 0;
		}
	);
	output.BufferData(results);
}
