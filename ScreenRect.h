#pragma once
#ifndef SCREEN_RECT_H
#define SCREEN_RECT_H

#include "glmHelp.h"

struct ScreenRect : public glm::vec4
{
	constexpr ScreenRect(const glm::vec4& vec);
	constexpr ScreenRect(const glm::vec2& topLeft, const glm::vec2& bottomRight);
	constexpr ScreenRect(const glm::vec2& topLeft, float w, float y);
	constexpr ScreenRect(float a, float b, float c, float d);
	inline constexpr bool Contains(const glm::vec2& point) const;
	inline constexpr bool Contains(float x, float y) const;
	inline constexpr bool Overlaps(const ScreenRect& other) const;
};

typedef ScreenRect Rect;

inline constexpr ScreenRect::ScreenRect(const glm::vec4& vec) : glm::vec4(vec)
{

}

inline constexpr ScreenRect::ScreenRect(const glm::vec2& topLeft, const glm::vec2& bottomRight) : glm::vec4(topLeft, bottomRight - topLeft)
{

}

inline constexpr ScreenRect::ScreenRect(const glm::vec2& topLeft, float w, float h) : glm::vec4(topLeft, w, h)
{

}

inline constexpr ScreenRect::ScreenRect(float x, float y, float w, float h) : glm::vec4(x, y, w, h)
{

}

inline constexpr bool ScreenRect::Contains(const glm::vec2& point) const
{
	return point.x > this->x && point.x < this->x + this->z &&
		point.y > this->y && point.y < this->y + this->w;
}

inline constexpr bool ScreenRect::Contains(float x, float y) const
{
	return this->Contains(glm::vec2(x, y));
}

inline constexpr bool ScreenRect::Overlaps(const ScreenRect& other) const
{
	return this->x < other.x + other.z && this->x + this->z > other.x &&
		this->y < other.y + other.w && this->y + this->w > other.y;
}

#endif // SCREEN_RECT_H

