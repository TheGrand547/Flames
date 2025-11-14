#pragma once
#ifndef RESOURCE_BANK_H
#define RESOURCE_BANK_H
#include "Log.h"
#include <format>
#include <unordered_map>

template<typename Type> struct Bank
{
private:
	inline static std::unordered_map<std::string_view, Type> elements;
public:
	static inline Type& Get(const std::string_view& name) noexcept
	{
		return ::Bank<Type>::elements[name];
	}

	static inline Type& Retrieve(const std::string_view& name) noexcept
	{
		return ::Bank<Type>::elements.at(name);
	}

	template<typename Data, typename Function> static inline void for_each(const Data& data, Function function)
	{
		for (const auto& element : data)
		{
			if (elements.contains(element))
			{
				function(elements[element]);
			}
			else
			{
				Log("'{}' is not a pre-existing element in this ResourceBank", element);
			}
		}
	}
};
#endif // RESOURCE_BANK_H