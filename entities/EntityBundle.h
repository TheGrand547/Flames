#pragma once
#ifndef ENTITY_BUNDLE_H
#define ENTITY_BUNDLE_H
#include <cstdint>
#include <ranges>

using IDType = std::uint32_t;

template<typename T> struct Bundle
{
	IDType id;
	T data;
	/*
	Bundle<T>& operator=(const Bundle<T>& other) noexcept = default;
	Bundle<T>& operator=(Bundle<T>&& other) noexcept = default;
	Bundle(Bundle<T>&& other) noexcept = default;
	Bundle(const Bundle<T>& other) noexcept = default;
	~Bundle() noexcept = default;
	*/
};

template<typename T> constexpr Bundle<T> to_bundle(IDType id, T element) noexcept
{
	return Bundle<T>(id, element);
}

#define BundleID   std::views::transform([](const auto& r) {return r.id;})
#define BundleData std::views::transform([](const auto& r) {return r.data;})

/*
template<typename... T> struct Bundle
{
	IDType id;
	std::tuple<T...> data;
};*/

#endif // ENTITY_BUNDLE_H