#pragma once
#ifndef RESOURCE_BANK_H
#define RESOURCE_BANK_H
#include "Shader.h"
#include "Buffer.h"
#include "VertexArray.h"
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
};

using ShaderBank = Bank<Shader>;
using VAOBank = Bank<VAO>;
using BufferBank = Bank<ArrayBuffer>;

#endif // RESOURCE_BANK_H