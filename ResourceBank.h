#pragma once
#ifndef RESOURCE_BANK_H
#define RESOURCE_BANK_H
#include "Shader.h"
#include "Buffer.h"
#include "VertexArray.h"
#include <unordered_map>

namespace ShaderBank
{
	Shader& Get(const std::string_view& name);
}

namespace VAOBank
{
	VAO& Get(const std::string_view& name);
}

namespace BufferBank
{
	ArrayBuffer& Get(const std::string_view& name);
}

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

#endif // RESOURCE_BANK_H