#pragma once
#ifndef BUFFER_SYNC_H
#define BUFFER_SYNC_H
#include <mutex>
#include <thread>

template<typename T>
struct BufferSync
{
protected:
	std::mutex mutex;
	T data;
public:
	using value_type = T;

	template <typename F> void ExclusiveOperation(F func)
	{
		std::lock_guard<std::mutex> lock(this->mutex);
		func(this->data);
	}

	void Swap(T& ref)
	{
		std::lock_guard<std::mutex> lock(this->mutex);
		std::swap(ref, this->data);
	}
};

#endif // BUFFER_SYNC_H