#include "MagneticAttack.h"
#include "Interpolation.h"

MagneticAttack::MagneticAttack(std::uint16_t growTime, std::uint16_t maxTime, std::uint16_t shrinkTime, float maxRadius) noexcept : 
	local(QuatIdentity()), ticksAlive(static_cast<std::uint16_t>(-1)), growTime(growTime), maxTime(maxTime + growTime), shrinkTime(shrinkTime + maxTime + growTime), radius(0.f),
	maxRadius(maxRadius)
{

}

void MagneticAttack::Configure(std::uint16_t growTime, std::uint16_t maxTime, std::uint16_t shrinkTime, float maxRadius) noexcept
{
	this->growTime   = growTime;
	this->maxTime    = this->growTime + maxTime;
	this->shrinkTime = this->maxTime + shrinkTime;
	this->maxRadius = maxRadius;
}

bool MagneticAttack::Finished() const noexcept
{
	return this->ticksAlive > shrinkTime;
}

glm::mat4 MagneticAttack::GetMatrix(const glm::vec3& position) const noexcept
{
	Model model(position, this->local, this->radius);
	return model.GetModelMatrix();
}

void MagneticAttack::Start(const Transform& transform) noexcept
{
	this->ticksAlive = 0;
	this->local = transform.rotation;
}

void MagneticAttack::Update() noexcept
{
	this->ticksAlive++;
	if (this->ticksAlive > this->shrinkTime)
	{
		// Dead
		this->radius = 0.f;
	}
	else if (this->ticksAlive > this->maxTime)
	{
		// Shrink
		std::uint16_t difference = this->ticksAlive - this->growTime;
		float ratio = static_cast<float>(difference) / (this->shrinkTime - this->maxTime);
		this->radius = Easing::lerp(this->maxRadius, 0.f, Easing::Circular(ratio));
	}
	else if (this->ticksAlive > this->growTime)
	{
		// Do nothing
		this->radius = this->maxRadius;
	}
	else
	{
		// Grow
		std::uint16_t difference = this->ticksAlive;
		float ratio = static_cast<float>(difference) / this->growTime;
		this->radius = Easing::lerp(0.f, this->maxRadius, Easing::Circular(ratio));
	}
	this->local = glm::angleAxis(glm::radians(1.25f), glm::normalize(glm::vec3(1.f, 2.f, 3.f))) * this->local;
}

Sphere MagneticAttack::GetCollision(const glm::vec3& center) const noexcept
{
	return Sphere(center, this->radius);
}
