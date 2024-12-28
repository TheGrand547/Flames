#pragma once
#ifndef ANIMATION_H
#define ANIMATION_H
#include "glmHelp.h"

// HACK
template<typename F> F EaseOut(F functor) noexcept
{
	return [](const double& delta) {return functor(1. - delta)};
}

template<class T, typename F>
class Animation
{
public:


	inline T Get(const double& delta) const noexcept
	{
		return glm::lerp(this->start, this->end, this->functor(delta))
	}

	inline T SphereGet(const double& delta) const noexcept
	{
		return glm::slerp(this->start, this->endTick, this->functor(delta));
	}



protected:
	T start, T end;
	F functor;
	// Probably overkill lol
	std::size_t startTick, endTick, duration;
};
#endif // ANIMATION_H