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

	void Fill(T value) noexcept
	{
		this->elements.fill(value);
		this->full = true;
		this->frontIndex = 0;
	}

	const std::array<T, N>& Get() const noexcept
	{
		return this->elements;
	}

	std::vector<T> GetLinear() const noexcept
	{
		std::vector<T> elements;
		const auto start = this->elements.cbegin() + this->frontIndex;
		elements.reserve((this->full) ? N : this->frontIndex);
		if (this->full)
		{
			std::copy(start, this->elements.end(), std::back_inserter(elements));
		}
		std::copy(this->elements.begin(), start, std::back_inserter(elements));

		return elements;
	}
};

#endif // CIRCULAR_BUFFER_H