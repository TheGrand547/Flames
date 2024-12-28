#pragma once
#ifndef INTERPOLATION_H
#define INTERPOLATION_H
#include "glmHelp.h"

// TODO: The rest of these things
/*
enum class Interploation
{
	Linear, Quadratic, Cubic, Quartic, Quintic, Sine, Cosine,
	EaseOutLinear, EaseOutQuadratic, EaseOutCubic, EaseOutQuartic, EaseOutQuintic, EaseOutSine, EaseOutCosine,
};
*/

namespace Interpolation
{
	template<typename T> inline T Linear(const T& start, const T& end, const double& delta) noexcept
	{
		return glm::lerp(start, end, delta);
	}

	template<typename T> inline T Quadratic(const T& start, const T& end, const double& delta) noexcept
	{
		return glm::lerp(start, end, delta * delta);
	}

	template<typename T> inline T Cubic(const T& start, const T& end, const double& delta) noexcept
	{
		return glm::lerp(start, end, delta * delta * delta);
	}

	template<typename T> inline T Quartic(const T& start, const T& end, const double& delta) noexcept
	{
		return glm::lerp(start, end, glm::pow(delta, 4));
	}

	template<typename T> inline T Quintic(const T& start, const T& end, const double& delta) noexcept
	{
		return glm::lerp(start, end, glm::pow(delta, 5));
	}
}
#endif // INTERPOLATION_H