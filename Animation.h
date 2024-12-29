#pragma once
#ifndef ANIMATION_H
#define ANIMATION_H
#include "glmHelp.h"
#include "Interpolation.h"
#include <tuple>

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

// TODO: The smarter way of doing it, where every "get" is assumed to be the next tick so less has to be stored and dealt with

template<class Type>
class SimpleAnimation
{
protected:
	Type start, end;
	EasingFunction easeIn, easeOut;
	// Probably overkill lol
	std::size_t startTick, inDuration, outDuration;
	bool finished;
public:
	void Start(const std::size_t& currentTick) noexcept
	{
		this->startTick = currentTick;
		this->finished = false;
	}

	inline Type Get(const std::size_t& currentTick) noexcept
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
			if constexpr (std::is_same_v<Type, glm::quat>)
			{
				return glm::slerp(this->end, this->start, static_cast<float>(this->easeOut(delta)));
			}
			else
			{
				return glm::lerp(this->end, this->start, static_cast<float>(this->easeOut(delta)));
			}
		}
		else
		{
			double delta = static_cast<double>(currentTick - this->startTick) / this->inDuration;
			if constexpr (std::is_same_v<Type, glm::quat>)
			{
				return glm::slerp(this->start, this->end, static_cast<float>(this->easeIn(delta)));
			}
			else
			{
				return glm::lerp(this->start, this->end, static_cast<float>(this->easeIn(delta)));
			}
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

	SimpleAnimation(const Type& start, const Type& end, std::size_t inDuration = 128, EasingFunction easeIn = Easing::Linear,
							std::size_t outDuration = 0, EasingFunction easeOut = Easing::Linear) noexcept : start(start), end(end),
		easeIn(easeIn), easeOut(easeOut), startTick(0), inDuration(inDuration), outDuration(outDuration), finished(false) { }

	SimpleAnimation(const Type& start, std::size_t inDuration, EasingFunction easeIn,
		const Type& end, std::size_t outDuration, EasingFunction easeOut) noexcept : start(start), end(end),
		easeIn(easeIn), easeOut(easeOut), startTick(0), inDuration(inDuration), outDuration(outDuration), finished(false) { }
};

template<std::size_t N, class T>
	requires requires
{
	N > 1;
}
class Animation
{
public:
	/*
	struct Element
	{
		T KeyFrame;
		std::size_t Duration;
		EasingFunction Easing;
	};*/
	typedef std::tuple<T, std::size_t, EasingFunction> Element;

	template<typename Failure>
	Animation(const T& base, const std::array<Failure, N - 1>& transitions) : keyFrames(), stages(), startTick(0), stageIndex(0), finished(false)
	{
		keyFrames[0] = base;
		for (std::size_t i = 1; i < N; i++)
		{
			keyFrames[i] = std::get<0>(transitions[i - 1]);
			stages[i - 1] = std::make_pair(std::get<1>(transitions[i - 1]), std::get<2>(transitions[i - 1]));
		}
	}

	void Start(const std::size_t& currentTick) noexcept
	{
		this->startTick = currentTick;
		this->stageIndex = 0;
		this->finished = false;
	}

	T Get(const std::size_t& currentTick) noexcept
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
			return this->keyFrames[this->stageIndex++];
		}
		return glm::lerp(this->keyFrames[this->stageIndex], this->keyFrames[this->stageIndex + 1], static_cast<float>(current.second(delta)));
	}

	inline bool IsFinished() const noexcept
	{
		return this->finished;
	}

protected:
	std::array<T, N> keyFrames;
	std::array<std::pair<std::size_t, EasingFunction>, N - 1> stages;
	std::size_t startTick, stageIndex;
	bool finished;
};

#endif // ANIMATION_H