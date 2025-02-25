#pragma once
#ifndef MAGNETIC_FIELD_ATTACK_H
#define MAGNETIC_FIELD_ATTACK_H
#include "glmHelp.h"
#include "Sphere.h"

// 'owned' by something else, only concerned with 
struct MagneticAttack
{
protected:
	glm::quat local;
	std::uint16_t ticksAlive;
	std::uint16_t growTime, maxTime, shrinkTime;
	float radius, maxRadius;
public:
	
	MagneticAttack(std::uint16_t growTime, std::uint16_t maxTime, std::uint16_t shrinkTime, float maxRadius) noexcept;

	bool Finished() const noexcept;

	glm::mat4 GetMatrix(const glm::vec3& position) const noexcept;

	void Configure(std::uint16_t growTime, std::uint16_t maxTime, std::uint16_t shrinkTime, float maxRadius) noexcept;
	void Start(const Transform& transform) noexcept;
	void Update() noexcept;

	Sphere GetCollision(const glm::vec3& center) const noexcept;
};

#endif // MAGNETIC_FIELD_ATTACK_H