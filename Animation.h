#pragma once
#ifndef ANIMATION_H
#define ANIMATION_H
#include "glmHelp.h"
#include "Interpolation.h"
#include "Transform.h"
#include <tuple>
#include <span>

// HACK
template<typename F> F EaseOut(F functor) noexcept
{
	return [](const double& delta) {return 1. - functor(1. - delta); };
}


template<typename T> concept Interpolator = requires(T func, const double& delta)
{
	{ func(delta) } -> std::convertible_to<double>;
};

typedef double (*EasingFunction)(const double&);

struct AnimationElement
{
	Transform KeyFrame;
	std::size_t Duration;
	EasingFunction Easing;
};

// TODO: The smarter way of doing it, where every "get" is assumed to be the next tick so less has to be stored and dealt with

template<typename InFunc = EasingFunction, typename OutFunc = EasingFunction>
	requires requires
{
	Interpolator<InFunc> and Interpolator<OutFunc>;
}
class SimpleAnimation
{
protected:
	Transform start, end;
	InFunc easeIn;
	OutFunc easeOut;
	// Probably overkill lol
	std::size_t startTick, inDuration, outDuration;
	bool finished;
public:
	void Start(const std::size_t& currentTick) noexcept
	{
		this->startTick = currentTick;
		this->finished = false;
	}

	inline Transform Get(const std::size_t& currentTick) noexcept
	{
		if (this->finished)
		{
			return this->start;
		}
		if (currentTick >= (this->startTick + this->inDuration))
		{
			std::size_t difference = currentTick - (this->startTick + this->inDuration);
			if (difference >= this->outDuration)
			{
				this->finished = true;
			}
			double delta = static_cast<double>(currentTick - (this->startTick + this->inDuration)) / this->outDuration;
			return Interpolation::lerp(this->end, this->start, this->easeOut(delta));
		}
		else
		{
			double delta = static_cast<double>(currentTick - this->startTick) / this->inDuration;
			return Interpolation::lerp(this->start, this->end, this->easeIn(delta));
		}
	}

	inline std::size_t Duration() const noexcept
	{
		return this->inDuration + this->outDuration;
	}

	inline bool IsFinished() const noexcept
	{
		return this->finished;
	}

	SimpleAnimation(const Transform& start, const Transform& end, std::size_t inDuration = 128, InFunc easeIn = Easing::Linear,
							std::size_t outDuration = 0, OutFunc easeOut = Easing::Linear) noexcept : start(start), end(end),
		easeIn(easeIn), easeOut(easeOut), startTick(0), inDuration(inDuration), outDuration(outDuration), finished(false) { }

	SimpleAnimation(const Transform& start, std::size_t inDuration, InFunc easeIn,
		const Transform& end, std::size_t outDuration, OutFunc easeOut) noexcept : start(start), end(end),
		easeIn(easeIn), easeOut(easeOut), startTick(0), inDuration(inDuration), outDuration(outDuration), finished(false) { }
};

template<std::size_t N>
	requires requires
{
	N > 1;
}
class Animation
{
public:
	
	Animation(const Transform& base, AnimationElement const (&transitions)[N - 1]) : keyFrames(), stages(), startTick(0), stageIndex(0), finished(false)
	{
		this->keyFrames[0] = base.Normalized();
		auto start = transitions;
		for (std::size_t i = 1; i < N; i++)
		{
			this->keyFrames[i] = start[i - 1].KeyFrame.Normalized();
			this->stages[i - 1] = std::make_pair(start[i - 1].Duration, start[i - 1].Easing);
		}
	}
	
	Animation(const Transform& base, const std::initializer_list<AnimationElement>& transitions) : keyFrames(), stages(), startTick(0), stageIndex(0), finished(false)
	{
		this->keyFrames[0] = base.Normalized();
		auto start = transitions.begin();
		for (std::size_t i = 1; i < N; i++)
		{
			this->keyFrames[i] = start[i - 1].KeyFrame.Normalized();
			this->stages[i - 1] = std::make_pair(start[i - 1].Duration, start[i - 1].Easing);
		}
	}

	void Start(const std::size_t& currentTick) noexcept
	{
		this->startTick = currentTick;
		this->stageIndex = 0;
		this->finished = false;
	}

	Transform Get(const std::size_t& currentTick) noexcept
	{
		if (this->finished || this->stageIndex >= (N - 1))
		{
			this->finished = true;
			return this->keyFrames[N - 1];
		}
		auto& current = this->stages[this->stageIndex];
		double delta = static_cast<double>(currentTick - this->startTick) / current.first;
		if (currentTick - this->startTick >= current.first)
		{
			this->startTick = currentTick;
			return this->keyFrames[++this->stageIndex];
		}
		return Interpolation::lerp(this->keyFrames[this->stageIndex], this->keyFrames[this->stageIndex + 1], current.second(delta));
	}

	inline bool IsFinished() const noexcept
	{
		return this->finished;
	}

protected:
	std::array<Transform, N> keyFrames;
	std::array<std::pair<std::size_t, EasingFunction>, N - 1> stages;
	std::size_t startTick, stageIndex;
	bool finished;
};

template<std::size_t N> Animation<N + 1> make_animation(const Transform& base, AnimationElement const (&transitions)[N])
{
	return Animation<N + 1>(base, transitions);
}

#endif // ANIMATION_H