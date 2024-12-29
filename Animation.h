#pragma once
#ifndef ANIMATION_H
#define ANIMATION_H
#include "glmHelp.h"
#include "Interpolation.h"

// HACK
template<typename F> F EaseOut(F functor) noexcept
{
	return [](const double& delta) {return 1. - functor(1. - delta); };
}

template<typename T> concept Interpolation = requires(T func, const double& delta)
{
	{ func(delta) } -> std::convertible_to<double>;
};

typedef double (*EasingFunction)(const double&);

template<class Type>
class SimpleAnimation
{
protected:
	Type start, end;
	EasingFunction easeIn, easeOut;
	// Probably overkill lol
	std::size_t startTick, inDuration, outDuration;
public:

	void Start(const std::size_t& currentTick) noexcept
	{
		this->startTick = currentTick;
	}

	inline Type Get(const std::size_t& currentTick) const noexcept
	{
		if (currentTick >= (this->startTick + this->inDuration))
		{
			double delta = static_cast<double>(currentTick - (this->startTick + this->inDuration)) / this->outDuration;
			if constexpr (std::is_same_v<Type, glm::quat>)
			{
				return glm::slerp(this->end, this->start, this->easeOut(delta));
			}
			else
			{
				return glm::lerp(this->end, this->start, static_cast<float>(this->easeOut(delta)));
			}
		}
		else
		{
			double delta = static_cast<double>(currentTick - this->startTick) / this->inDuration;
			if constexpr (std::is_same_v < Type, glm::quat>)
			{
				return glm::slerp(this->start, this->end, this->easeIn(delta));
			}
			else
			{
				return glm::lerp(this->start, this->end, static_cast<float>(this->easeIn(delta)));
			}
		}
	}

	const std::size_t Duration() const noexcept
	{
		return this->inDuration + this->outDuration;
	}

	SimpleAnimation(const Type& start, const Type& end, std::size_t inDuration = 128, EasingFunction easeIn = Easing::Linear,
							std::size_t outDuration = 0, EasingFunction easeOut = Easing::Linear) noexcept : start(start), end(end),
		easeIn(easeIn), easeOut(easeOut), startTick(0), inDuration(inDuration), outDuration(outDuration) { }

	SimpleAnimation(const Type& start, std::size_t inDuration, EasingFunction easeIn,
		const Type& end, std::size_t outDuration, EasingFunction easeOut) noexcept : start(start), end(end),
		easeIn(easeIn), easeOut(easeOut), startTick(0), inDuration(inDuration), outDuration(outDuration) {
	}
};

template<class T, typename F, std::size_t N>
	requires requires
{
	Interpolation<F>;
	N > 1;
}
class Animation
{
public:
	Animation(const std::array<T, N>& keyFrames, std::array<T, N - 1>) : keyFrames(keyFrames) {}

	void Start(const std::size_t& currentTick) noexcept
	{
		this->startFrame = currentTick;
		this->stageIndex = 0;
	}

protected:
	std::array<T, N> keyFrames;
	std::array<std::pair<F, std::size_t>, N - 1> stages;
	std::size_t startFrame, stageIndex;
};

#endif // ANIMATION_H