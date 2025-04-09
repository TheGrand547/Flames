#pragma once
#ifndef STATIC_VECTOR_H
#define STATIC_VECTOR_H

#include <memory>
#include <execution>

template<typename Type>
class StaticVector
{
private:
	std::unique_ptr<Type[]> pointer;
	std::size_t length;
public:
	StaticVector() noexcept = default;
	StaticVector(std::size_t size) noexcept : pointer(std::make_unique<Type[]>(size)), length(size)
	{
		std::uninitialized_default_construct(this->begin(), this->end());
	}
	
	template<typename Policy>
	StaticVector(std::size_t size, Policy executionPolicy) noexcept : pointer(std::make_unique<Type[]>(size)), length(size)
	{
		std::uninitialized_default_construct(executionPolicy, this->begin(), this->end());
	}

	StaticVector(std::size_t size, Type member) noexcept : pointer(std::make_unique<Type[]>(size)), length(size)
	{
		std::uninitialized_fill(this->begin(), this->end(), member);
	}

	template<typename Policy>
	StaticVector(std::size_t size, Type member, Policy executionPolicy) noexcept : pointer(std::make_unique<Type[]>(size)), length(size)
	{
		std::uninitialized_fill(executionPolicy, this->begin(), this->end(), member);
	}

	inline Type& operator[](std::size_t index) noexcept
	{
		return this->pointer[index];
	}

	inline const Type& operator[](std::size_t index) const noexcept
	{
		return this->pointer[index];
	}

	inline std::size_t size() const noexcept
	{
		return this->length;
	}

	inline Type* begin() noexcept
	{
		return this->pointer.get();
	}

	inline const Type* cbegin() const noexcept
	{
		return this->pointer.get();
	}

	inline Type* end() noexcept
	{
		return this->pointer.get() + this->length;
	}

	inline Type* cend() const noexcept
	{
		return this->pointer.get() + this->length;
	}
};

#endif // STATIC_VECTOR_H