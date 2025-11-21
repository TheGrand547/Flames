#include "DecayLight.h"
#include <array>

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
	const float n1 = 7.5625;
	const float d1 = 2.75;

	if (t < (1 / d1)) {
		return n1 * t * t;
	}
	else if (t < (2 / d1)) {
		return n1 * (t -= 1.5 / d1) * t + 0.75;
	}
	else if (t < (2.5 / d1)) {
		return n1 * (t -= 2.25 / d1) * t + 0.9375;
	}
	else {
		return n1 * (t -= 2.625 / d1) * t + 0.984375;
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
	auto index = static_cast<unsigned int>(glm::ceil(left / total * colorPhases.size()));

	float progress = bouncy(left / total);

	result.position  = glm::vec4(this->position, MaxRadius * progress);
	result.direction = glm::vec4(1.f);
	result.color     = glm::vec4(colorPhases[index % colorPhases.size()], 1.f);
	result.constants = glm::vec4(1.f, 0.2f, 0.0f, 1.f);
	return result;
}
