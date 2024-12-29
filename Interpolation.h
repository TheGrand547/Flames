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

namespace Easing
{
	inline double Linear(const double& delta) noexcept
	{
		return delta;
	}

	inline double Quadratic(const double& delta) noexcept
	{
		return delta * delta;
	}

	inline double Cubic(const double& delta) noexcept
	{
		return glm::pow(delta, 3);
	}

	inline double Quartic(const double& delta) noexcept
	{
		return glm::pow(delta, 4);
	}

	inline double Quintic(const double& delta) noexcept
	{
		return glm::pow(delta, 5);
	}

	template<double X> inline double Power(const double& delta) noexcept
	{
		return glm::pow(delta, X);
	}

	template<double X> inline double EaseOutPower(const double& delta) noexcept
	{
		return 1. - glm::pow(1. - delta, X);
	}


	inline double EaseOutLinear(const double& delta) noexcept
	{
		return 1. - delta;
	}

	inline double EaseOutQuadratic(const double& delta) noexcept
	{
		return EaseOutPower<2.>(delta);
	}

	inline double EaseOutCubic(const double& delta) noexcept
	{
		return EaseOutPower<3.>(delta);
	}

	inline double EaseOutQuartic(const double& delta) noexcept
	{
		return EaseOutPower<4.>(delta);
	}

	inline double EaseOutQuintic(const double& delta) noexcept
	{
		return EaseOutPower<5.>(delta);
	}
}
#endif // INTERPOLATION_H