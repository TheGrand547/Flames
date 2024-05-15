#pragma once
#ifndef LINES_H
#define LINES_H
#include <vector>
#include "glmHelp.h"
#include "glm/gtx/norm.hpp"
#include "util.h"

class Plane;
struct Ray;
struct LineSegment;

struct LineBase
{
	float Distance(const LineBase& line) const noexcept;
	float Distance(const LineBase& line, glm::vec3& thisPoint, glm::vec3& otherPoint) const noexcept;

	virtual glm::vec3 PointA() const noexcept = 0;
	virtual glm::vec3 PointB() const noexcept = 0;
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
	
	inline Line() noexcept = default;
	inline Line(const glm::vec3& point, const glm::vec3& dir) noexcept;

	inline bool operator==(const Line& other) const noexcept = default;
	inline bool operator!=(const Line& other) const noexcept = default;
	inline bool operator<(const Line& other) const noexcept = delete;
	inline bool operator<=(const Line& other) const noexcept = delete;
	inline bool operator>(const Line& other) const noexcept = delete;
	inline bool operator>=(const Line& other) const noexcept = delete;

	Line operator*(const Line& other) = delete;
	Line operator/(const Line& other) = delete;
	Line operator+(const Line& other) = delete;
	Line operator-(const Line& other) = delete;

	virtual glm::vec3 PointClosestTo(const glm::vec3& point) const noexcept override;
	virtual glm::vec3 PointA() const noexcept;
	virtual glm::vec3 PointB() const noexcept;
};

inline Line::Line(const glm::vec3& point, const glm::vec3& dir) noexcept : point(point), dir(glm::normalize(dir)) {}


struct Ray : public Line
{
	inline Ray() noexcept = default;
	inline Ray(const glm::vec3& a, const glm::vec3& b) noexcept;
	inline Ray(const Ray& other) noexcept;

	inline bool operator==(const Ray& other) const noexcept = default;
	inline bool operator!=(const Ray& other) const noexcept = default;
	
	virtual glm::vec3 PointClosestTo(const glm::vec3& point) const noexcept override;
};

inline Ray::Ray(const glm::vec3& a, const glm::vec3& b) noexcept : Line(a, b) {}

inline Ray::Ray(const Ray& other) noexcept : Line(other.point, other.dir) {}

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
	inline LineSegment(const glm::vec3& a = glm::vec3(0.f), const glm::vec3& b = glm::vec3(1.f)) noexcept : A(a), B(b) {}
	inline LineSegment(const LineSegment& other) noexcept : A(other.A), B(other.B) {}
	inline LineSegment(const LineSegment&& other) noexcept : A(other.A), B(other.B) {}
	inline LineSegment& operator=(const LineSegment& other) noexcept;
	inline LineSegment& operator=(const LineSegment&& other) noexcept;

	inline bool operator==(const LineSegment& other) const noexcept { return this->A == other.A && this->B == other.B; }
	inline bool operator!=(const LineSegment& other) const noexcept { return this->A != other.A || this->B != other.B; }

	float Length() const noexcept;
	float Magnitude() const noexcept;
	float SquaredLength() const noexcept;

	inline glm::vec3 MidPoint() const noexcept;
	
	inline glm::vec3 Lerp(float t) const noexcept;

	inline glm::vec3 Direction() const noexcept;
	inline glm::vec3 UnitDirection() const noexcept;

	std::vector<LineSegment> Split(const Plane& plane) const;

	virtual glm::vec3 PointA() const noexcept;
	virtual glm::vec3 PointB() const noexcept;
	virtual glm::vec3 PointClosestTo(const glm::vec3& point) const noexcept;
};

inline LineSegment& LineSegment::operator=(const LineSegment& other) noexcept
{
	this->A = other.A;
	this->B = other.B;
	return *this;
}

inline LineSegment& LineSegment::operator=(const LineSegment&& other) noexcept
{
	this->A = other.A;
	this->B = other.B;
	return *this;
}


inline glm::vec3 LineSegment::Direction() const noexcept
{
	return this->B - this->A;
}

inline glm::vec3 LineSegment::UnitDirection() const noexcept
{
	return glm::normalize(this->B - this->A);
}


inline glm::vec3 LineSegment::Lerp(float t) const noexcept
{
	return this->A + (this->B - this->A) * t;
}

inline glm::vec3 LineSegment::PointA() const noexcept
{
	return this->A;
}

inline glm::vec3 LineSegment::PointB() const noexcept
{
	return this->B;
}

inline glm::vec3 LineSegment::MidPoint() const noexcept
{
	return (this->A + this->B) / 2.f;
}

#endif // LINES_H