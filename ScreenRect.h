#pragma once
#ifndef SCREEN_RECT_H
#define SCREEN_RECT_H

#include "glmHelp.h"

struct ScreenRect : public glm::vec4
{
	inline ScreenRect(const glm::vec4& vec) noexcept;
	inline ScreenRect(const glm::vec2& topLeft, const glm::vec2& bottomRight) noexcept;
	inline ScreenRect(const glm::vec2& topLeft, float w, float y) noexcept;
	inline ScreenRect(float a, float b, float c, float d) noexcept;
	inline ScreenRect& operator=(const ScreenRect& other) noexcept = default;
	inline ScreenRect& operator=(const glm::vec4& other) noexcept
	{
		this->data = other.data;
	}
	inline bool Contains(const glm::vec2& point) const noexcept;
	inline bool Contains(float x, float y) const noexcept;
	inline bool Overlaps(const ScreenRect& other) const noexcept;
};

typedef ScreenRect Rect;

inline ScreenRect::ScreenRect(const glm::vec4& vec) noexcept : glm::vec4(vec)
{

}

inline ScreenRect::ScreenRect(const glm::vec2& topLeft, const glm::vec2& bottomRight) noexcept : glm::vec4(topLeft, bottomRight - topLeft)
{

}

inline ScreenRect::ScreenRect(const glm::vec2& topLeft, float w, float h) noexcept : glm::vec4(topLeft, w, h)
{

}

inline ScreenRect::ScreenRect(float x, float y, float w, float h) noexcept : glm::vec4(x, y, w, h)
{

}

inline bool ScreenRect::Contains(const glm::vec2& point) const noexcept
{
	return point.x > this->x && point.x < this->x + this->z &&
		point.y > this->y && point.y < this->y + this->w;
}

inline bool ScreenRect::Contains(float x, float y) const noexcept
{
	return this->Contains(glm::vec2(x, y));
}

inline bool ScreenRect::Overlaps(const ScreenRect& other) const noexcept
{
	return this->x < other.x + other.z && this->x + this->z > other.x &&
		this->y < other.y + other.w && this->y + this->w > other.y;
}

#endif // SCREEN_RECT_H

