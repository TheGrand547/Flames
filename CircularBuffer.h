#pragma once
#ifndef CIRCULAR_BUFFER_H
#define CIRCULAR_BUFFER_H
#include <array>
#include <cassert>


template<typename T, std::size_t N>
struct CircularBuffer
{
protected:
	std::array<T, N> elements{};
	bool full = false;
	std::size_t frontIndex = 0;
public:
	CircularBuffer() noexcept = default;

	bool IsFull() const noexcept
	{
		return this->full;
	}

	std::size_t length() const noexcept
	{
		return (this->full) ? N : this->frontIndex;
	}

	T& Bottom() const noexcept
	{
		
	}

	T& Element(std::size_t index) const noexcept
	{
		return this->elements[index];
	}

	void Push(T element) noexcept
	{
		this->elements[this->frontIndex++] = element;
		if (this->frontIndex >= N)
		{
			this->full = true;
			this->frontIndex = 0;
		}
	}

	T PushPop(T element) noexcept
	{
		assert(this->full);
		T local = this->elements[this->frontIndex];
		this->elements[this->frontIndex++] = element;
		if (this->frontIndex >= N)
		{
			this->full = true;
			this->frontIndex = 0;
		}
		return local;
	}

	void Reset() noexcept
	{
		this->elements.fill({});
		this->full = false;
		this->frontIndex = 0;
	}
};

#endif // CIRCULAR_BUFFER_H