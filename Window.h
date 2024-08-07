#pragma once
#ifndef WINDOW_H
#define WINDOW_H
#include <glew.h>
#include <glm/glm.hpp>
#include <glm/ext/matrix_clip_space.hpp>

namespace Window
{
	static GLsizei Height = 1000, Width = 1000;
	static float FOV = 70.f;
	static float AspectRatio = 1.f;

	inline glm::ivec2 GetSize() noexcept
	{
		return glm::ivec2(Width, Height);
	}

	inline glm::vec2 GetSizeF() noexcept
	{
		return glm::vec2(Width, Height);
	}

	constexpr void Update(GLsizei width, GLsizei height) noexcept
	{
		Width = width;
		Height = height;
		AspectRatio = static_cast<float>(width) / static_cast<float>(height);
	}

	inline float GetYFOV() noexcept
	{
		return glm::radians(FOV * AspectRatio);
	}

	inline glm::mat4 GetPerspective(float zNear, float zFar)
	{
		return glm::perspective(GetYFOV(), AspectRatio, zNear, zFar);
	}

	inline glm::mat4 GetOrthogonal() noexcept
	{
		return glm::ortho<float>(0.f, static_cast<float>(Width), static_cast<float>(Height), 0.f);
	}

	inline void Viewport() noexcept
	{
		glViewport(0, 0, Width, Height);
	}
};

#endif // WINDOW_H
