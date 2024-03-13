#pragma once
#ifndef GLM_HELP_H
#define GLM_HELP_H
#include <format>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <iostream>
#include <limits>

#define RotateX(matrix, radians) glm::rotate(matrix, radians, glm::vec3(1.f, 0.f, 0.f))
#define RotationX(radians) RotateX(glm::mat4(1.0f), radians)

#define RotateY(matrix, radians) glm::rotate(matrix, radians, glm::vec3(0.f, 1.f, 0.f))
#define RotationY(radians) RotateY(glm::mat4(1.0f), radians)

#define RotateZ(matrix, radians) glm::rotate(matrix, radians, glm::vec3(0.f, 0.f, 1.f))
#define RotationZ(radians) RotateZ(glm::mat4(1.0f), radians)

#define Vec4to3(vec4) glm::vec3(vec4.x, vec4.y, vec4.z)
#define Vec3to4(vec3) glm::vec4(vec3.x, vec3.y, vec3.z, 1.f)

#define absdot(x, y) glm::abs(glm::dot(x, y))

namespace Constexpr

{
	// Sourced from https://stackoverflow.com/questions/8622256/in-c11-is-sqrt-defined-as-constexpr/34134071#34134071
	template<typename T = double>
	T constexpr sqrtNewtonRaphson(T x, T curr, T prev)
	{
		return curr == prev
			? curr
			: sqrtNewtonRaphson<T>(x, static_cast<T>(0.5) * (curr + x / curr), curr);
	}

	template<typename T = double>
	inline T constexpr sqrt(T x)
	{
		return x >= 0 && x < std::numeric_limits<T>::infinity()
			? sqrtNewtonRaphson<T>(x, x, static_cast<T>(0))
			: std::numeric_limits<T>::quiet_NaN();
	}


	template<::glm::length_t T, typename W, ::glm::qualifier Q>
	inline constexpr W dot(const ::glm::vec<T, W, Q>& a, const ::glm::vec<T, W, Q>& b)
	{
		if (std::is_constant_evaluated())
		{
			W total{};
			for (glm::length_t i = 0; i < T; i++)
			{
				total += a[i] * b[i];
			}
			return total;
		}
		else
		{
			return glm::dot(a, b);
		}
	}

	template<::glm::length_t T, typename W, ::glm::qualifier Q>
	inline constexpr W length(const ::glm::vec<T, W, Q>& a)
	{
		if (std::is_constant_evaluated())
		{
			return ::Constexpr::sqrt<W>(::Constexpr::dot(a, a));
		}
		else
		{
			return glm::length(a);
		}
	}

	template<::glm::length_t T, typename W, ::glm::qualifier Q>
	inline constexpr ::glm::vec<T, W, Q> normalize(const ::glm::vec<T, W, Q>& a)
	{
		if (std::is_constant_evaluated())
		{
			return a / ::Constexpr::length(a);
		}
		else
		{
			return glm::normalize(a);
		}
	}
}

template<glm::length_t T, typename W, glm::qualifier Q>
std::ostream& operator<<(std::ostream& os, const glm::vec<T, W, Q>& vec)
{
	os << "(";
	for (glm::length_t i = 0; i < T; i++)
	{
		os << vec[i];
		if (i != T - 1)
			os << ", ";
	}
	os << ")";
	return os;
}
/*
template <> template<glm::length_t T, typename W, glm::qualifier Q>
struct std::formatter<const glm::vec<T, W, Q>&> : std::formatter<std::string> {
	auto format(const glm::vec<T, W, Q>& vec, format_context& ctx) const {
		return formatter<string>::format(
			std::format("({}, {}, {})", vec), ctx);
	}
};*/

//  SLOPPY
inline std::ostream& operator<<(std::ostream& os, const glm::quat& vec)
{
	os << "(";
	for (glm::length_t i = 0; i < 4; i++)
	{
		os << vec[i];
		if (i != 4 - 1)
			os << ", ";
	}
	os << ")";
	return os;
}

template<typename T> concept IsVec3 = std::same_as<std::remove_cvref_t<T>, glm::vec3>;

namespace glm 
{
	// Returns the absolute value of the dot product of x on y
	template<length_t L, typename T, qualifier Q> inline T adot(vec<L, T, Q> const& x, vec<L, T, Q> const& y)
	{
		return glm::abs(glm::dot(x, y));
	}
}

#endif // GLM_HELP_H