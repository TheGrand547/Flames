#pragma once
#ifndef BUFFER_SYNC_H
#define BUFFER_SYNC_H
#include <mutex>
#include <semaphore>
#include <thread>

template<typename T>
struct BufferSync
{
protected:
	// This feels like a complete hack
	mutable std::mutex mutex;
	// Investigate this, so it'll count up on use
	//mutable std::binary_semaphore mutex{ 1 };
	T data;
public:
	using value_type = T;

	template <typename F> inline auto ExclusiveOperation(F func) noexcept
	{
		std::lock_guard<std::mutex> lock(this->mutex);
		return func(this->data);
	}

	template <typename F> inline auto ExclusiveOperation(F func) const noexcept
	{
		std::lock_guard<std::mutex> lock(this->mutex);
		return func(this->data);
	}

	inline void Swap(T& ref) noexcept
	{
		std::lock_guard<std::mutex> lock(this->mutex);
		std::swap(ref, this->data);
	}
	inline void Swap(T&& ref) noexcept
	{
		std::lock_guard<std::mutex> lock(this->mutex);
		this->data = std::move(ref);
	}
};

#endif // BUFFER_SYNC_H