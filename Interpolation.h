#pragma once
#ifndef INTERPOLATION_H
#define INTERPOLATION_H
#include "glmHelp.h"
#include <glm/gtx/compatibility.hpp>
#include "Transform.h"

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
	template<typename T> inline T lerp(const T& a, const T& b, const double& t)
	{
		return static_cast<T>(a + (b - a) * t);
	}

	template<typename T> inline T lerpStable(const T& a, const T& b, const double& t)
	{
		return static_cast<T>((1. - t) * a + t * b);
	}

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
		return std::pow(delta, 3);
	}

	inline double Quartic(const double& delta) noexcept
	{
		return std::pow(delta, 4);
	}

	inline double Quintic(const double& delta) noexcept
	{
		return std::pow(delta, 5);
	}

	template<double X> inline double Power(const double& delta) noexcept
	{
		return std::pow(delta, X);
	}

	template<double X> inline double EaseOutPower(const double& delta) noexcept
	{
		return 1. - std::pow(1. - delta, X);
	}

	inline double Circular(const double& delta) noexcept
	{
		return 1. - std::sqrt(1. - delta * delta);
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

	inline double EaseOutCircular(const double& delta) noexcept
	{
		return std::sqrt(1. - std::pow(delta - 1., 2.));
	}
}

namespace Interpolation
{
	inline Transform lerp(const Transform& start, const Transform& end, const double& delta) noexcept
	{
		return Transform(glm::lerp(start.position, end.position, static_cast<float>(delta)),
			glm::slerp(start.rotation, end.rotation, static_cast<float>(delta)));
	}
}

#endif // INTERPOLATION_H