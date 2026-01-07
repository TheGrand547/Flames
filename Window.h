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
	static float AspectRatio = static_cast<float>(Width) / Height;

	inline glm::ivec2 GetSize() noexcept
	{
		return glm::ivec2(Width, Height);
	}

	inline glm::vec2 GetSizeF() noexcept
	{
		return glm::vec2(Width, Height);
	}

	inline glm::vec2 GetHalfF() noexcept
	{
		return glm::vec2(static_cast<float>(Width / 2), static_cast<float>(Height / 2));
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
	glm::mat4 GetMatInternal(float zNear, float zFar);
	inline glm::mat4 GetPerspective(float zNear, float zFar)
	{
		//return glm::perspective(GetYFOV(), AspectRatio, zFar, zNear);
		//return glm::infinitePerspective(GetYFOV(), AspectRatio, zNear);
		return GetMatInternal(zNear, zFar);
	}

	inline glm::mat4 GetOrthogonal() noexcept
	{
		return glm::ortho<float>(0.f, static_cast<float>(Width), static_cast<float>(Height), 0.f);
	}

	inline void Viewport() noexcept
	{
		glViewport(0, 0, Width, Height);
	}

	inline glm::vec4 GetRect() noexcept
	{
		return glm::vec4(glm::vec2(0.f), ::Window::GetSizeF());
	}
};

#endif // WINDOW_H
