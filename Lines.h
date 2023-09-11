#pragma once
#ifndef LINES_H
#define LINES_H
#include "glmHelp.h"
#include "glm/gtx/norm.hpp"
#include "util.h"

struct Ray;
struct LineSgement;

struct LineBase
{
	float Distance(const LineBase& line) const noexcept;
	float Distance(const LineBase& line, glm::vec3& thisPoint, glm::vec3& otherPoint) const noexcept;

	virtual constexpr glm::vec3 PointA() const noexcept = 0;
	virtual constexpr glm::vec3 PointB() const noexcept = 0;
	virtual glm::vec3 PointClosestTo(const glm::vec3& point) const noexcept = 0;
};

struct Line : public LineBase
{
	union
	{
		glm::vec3 point, initial;
	};
	union
	{
		glm::vec3 dir, direction, delta;
	};
	
	Line() = default;
	Line(const glm::vec3& point, const glm::vec3& dir);

	constexpr bool operator==(const Line& other) const = default;
	constexpr bool operator!=(const Line& other) const = default;
	bool operator<(const Line& other) const = delete;
	bool operator<=(const Line& other) const = delete;
	bool operator>(const Line& other) const = delete;
	bool operator>=(const Line& other) const = delete;

	Line operator*(const Line& other) = delete;
	Line operator/(const Line& other) = delete;
	Line operator+(const Line& other) = delete;
	Line operator-(const Line& other) = delete;

	virtual glm::vec3 PointClosestTo(const glm::vec3& point) const noexcept override;
	virtual constexpr glm::vec3 PointA() const noexcept;
	virtual constexpr glm::vec3 PointB() const noexcept;
};

constexpr glm::vec3 Line::PointA() const noexcept
{
	return this->initial;
}

constexpr glm::vec3 Line::PointB() const noexcept
{
	return this->initial + dir;
}

struct Ray : public Line
{
	Ray() = default;
	Ray(const glm::vec3& a, const glm::vec3& b);

	bool operator==(const Ray& other) const = default;
	bool operator!=(const Ray& other) const = default;
	
	virtual glm::vec3 PointClosestTo(const glm::vec3& point) const noexcept override;
};

struct LineSegment : public LineBase
{
	union
	{
		glm::vec3 pointA, point1, start, A;
	};
	union
	{
		glm::vec3 pointB, point2, end, B;
	};
	LineSegment() = default;
	constexpr LineSegment(const glm::vec3& a, const glm::vec3& b) : A(a), B(b) {}
	constexpr LineSegment(const LineSegment& other) noexcept : A(other.A), B(other.B) {}
	constexpr LineSegment(const LineSegment&& other) noexcept : A(other.A), B(other.B) {}
	constexpr LineSegment& operator=(const LineSegment& other) noexcept;
	constexpr LineSegment& operator=(const LineSegment&& other) noexcept;

	bool operator==(const LineSegment& other) const = default;
	bool operator!=(const LineSegment& other) const = default;

	float Length() const noexcept;
	float Magnitude() const noexcept;
	float SquaredLength() const noexcept;

	glm::vec3 MidPoint() const noexcept;
	
	inline constexpr glm::vec3 Lerp(float t) const noexcept;

	inline constexpr glm::vec3 Direction() const noexcept;

	virtual constexpr glm::vec3 PointA() const noexcept;
	virtual constexpr glm::vec3 PointB() const noexcept;
	virtual glm::vec3 PointClosestTo(const glm::vec3& point) const noexcept;
};

constexpr LineSegment& LineSegment::operator=(const LineSegment& other) noexcept
{
	this->A = other.A;
	this->B = other.B;
	return *this;
}

constexpr LineSegment& LineSegment::operator=(const LineSegment&& other) noexcept
{
	this->A = other.A;
	this->B = other.B;
	return *this;
}


inline constexpr glm::vec3 LineSegment::Direction() const noexcept
{
	return this->B - this->A;
}

inline constexpr glm::vec3 LineSegment::Lerp(float t) const noexcept
{
	return this->A + (this->B - this->A) * t;
}

constexpr glm::vec3 LineSegment::PointA() const noexcept
{
	return this->A;
}

constexpr glm::vec3 LineSegment::PointB() const noexcept
{
	return this->B;
}

#endif // LINES_H