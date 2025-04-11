#pragma once
#ifndef STATIC_VECTOR_H
#define STATIC_VECTOR_H

#include <memory>
#include <execution>
#include <span>

template<typename Type>
class StaticVector
{
private:
	std::unique_ptr<Type[]> pointer;
	std::size_t length;
public:
	using value_type = Type;

	inline StaticVector() noexcept : pointer(nullptr), length(0) {}
	StaticVector(std::size_t size) noexcept : pointer(std::make_unique_for_overwrite<Type[]>(size)), length(size)
	{
		std::uninitialized_default_construct(this->begin(), this->end());
	}
	
	template<typename Policy>
	StaticVector(std::size_t size, Policy executionPolicy) noexcept : pointer(std::make_unique_for_overwrite<Type[]>(size)), length(size)
	{
		std::uninitialized_default_construct(executionPolicy, this->begin(), this->end());
	}

	StaticVector(std::size_t size, Type member) noexcept : pointer(std::make_unique_for_overwrite<Type[]>(size)), length(size)
	{
		std::uninitialized_fill(this->begin(), this->end(), member);
	}

	template<typename Policy>
	StaticVector(std::size_t size, Type member, Policy executionPolicy) noexcept : pointer(std::make_unique_for_overwrite<Type[]>(size)), length(size)
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

	inline void make(std::span<Type> data) noexcept
	{
		this->length = data.size();
		this->pointer = std::make_unique_for_overwrite<Type[]>(data.size());
		std::memcpy(this->pointer.get(), data.data(), data.size_bytes());
	}

	inline void reserve(std::size_t size) noexcept
	{
		this->pointer = std::make_unique_for_overwrite<Type[]>(size);
	}
};

#endif // STATIC_VECTOR_H