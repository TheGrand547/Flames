#include "DecayLight.h"
#include <array>
#include "../util.h"

const static auto colorPhases = std::to_array(
	{
		glm::vec3(1.f, 0.f, 0.f),
		glm::vec3(1.f, 0.25f, 0.f),
		glm::vec3(1.f, 0.75f, 0.f),
		glm::vec3(0.9f, 0.9f, 0.f)
	}
);

static constexpr float MaxRadius = 3.5f;

DecayLight::DecayLight(const glm::vec3& position, std::uint16_t lifetime) : position(position), lifetime(lifetime), 
						timeLeft(lifetime)
{

}

// From: https://easings.net/#easeOutBounce
static float bouncy(float t)
{
	constexpr float n1 = 7.5625f;
	constexpr float d1 = 2.75f;

	constexpr float d1inv = 1.f / d1;

	if (t < d1inv) {
		return n1 * t * t;
	}
	else if (t < 2.f * d1inv) {
		t -= 1.5f * d1inv;
		return n1 * t * t + 0.75f;
	}
	else if (t < 2.5f * d1inv) {
		t -= 2.25f * d1inv;
		return n1 * t * t + 0.9375f;
	}
	else 
	{
		t -= 2.625f * d1inv;
		return n1 * t * t + 0.984375f;
	}
}


LightVolume DecayLight::Tick() noexcept
{
	LightVolume result;
	if (this->timeLeft > 0)
	{
		this->timeLeft--;
	}
	float left  = static_cast<float>(this->timeLeft);
	float total = static_cast<float>(this->lifetime);
	auto index  = static_cast<unsigned int>(glm::ceil(left / total * colorPhases.size()));

	float progress = bouncy(left / total);

	result.position  = glm::vec4(this->position, glm::max(MaxRadius * progress, EPSILON));
	result.direction = glm::vec4(1.f);
	result.color     = glm::vec4(colorPhases[index % colorPhases.size()], 1.f);
	result.constants = glm::vec4(1.f, 0.2f, 0.0f, 1.f);
	return result;
}
