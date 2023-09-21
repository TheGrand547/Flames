#include "OrientedBoundingBox.h"
#include <algorithm>
#include <bit>

OrientedBoundingBox::OrientedBoundingBox(const glm::vec3& euler, const glm::vec3& deltas) : matrix(1.f), halfs(glm::abs(deltas))
{
	this->Rotate(euler);
}

OrientedBoundingBox::OrientedBoundingBox(const Model& model) : OrientedBoundingBox(model.rotation, model.scale)
{
	this->matrix[3] = glm::vec4(model.translation, 1);
}

bool OrientedBoundingBox::Intersection(const Plane& plane, Collision& collision) const
{
	float delta = plane.Facing(this->Center());
	collision.normal = plane.GetNormal();

	// Ensure that the box can always go from out to inbounds
	if (!plane.TwoSided() && (delta < 0 || delta > glm::length(this->halfs)))
		return false;

	float projected = 0.f;

	for (glm::length_t i = 0; i < 3; i++)
		projected += glm::abs(glm::dot(glm::vec3(this->matrix[i] * this->halfs[i]), plane.GetNormal()));

	collision.distance = projected - glm::abs(delta);
	collision.point = this->Center() + glm::sign(delta) * glm::abs(collision.distance) * collision.normal; // This might be wrong?
	return glm::abs(projected) > glm::abs(delta);
}

bool OrientedBoundingBox::Overlap(const Capsule& other) const
{
	Collision collision{};
	return this->Overlap(other, collision);
}

bool OrientedBoundingBox::Overlap(const Capsule& other, Collision& collide) const
{
	return this->Overlap(Sphere(other.GetRadius(), other.ClosestPoint(this->Center())), collide);
}

// TODO: Standarize what the collision.point thingies mean
bool OrientedBoundingBox::Overlap(const Sphere& other, Collision& collision) const
{
	AABB local(this->halfs * 2.f);
	glm::vec3 transformed = this->WorldToLocal(other.center - this->Center());
	Sphere temp{ other.radius, transformed };
	bool result = local.Overlap(temp, collision);
	collision.normal = this->matrix * glm::vec4(collision.normal, 0);
	collision.point  = other.center + collision.normal * collision.depth;
	return result;
}

bool OrientedBoundingBox::OverlapCompleteResponse(const OrientedBoundingBox& other)
{
	Collision collide;
	if (this->Overlap(other, collide))
	{
		std::array<LineSegment, 12> axes = this->GetLineSegments(), fumoBox = other.GetLineSegments();
		glm::vec3 rotationPoint = glm::vec3(0, 0, 0), rotationAxis = glm::vec3(2, 1, 0);
		float mostOverlap = -INFINITY;

		struct rotation_help
		{
			float overlap = 0.f;
			glm::vec3 axis{ 2.f, 1.f, 0.f }, total{ 0.f };
			unsigned char count = 0;
		};

		std::array<rotation_help, 3> myAxisStruct{}, otherAxisStruct{};

		for (std::size_t i = 0; i < 12; i++)
		{
			std::size_t index = i % 3;
			auto lambda = [](LineSegment& line, const OBB& target, rotation_help& local)
				{
					Collision nearIntersection{}, farIntersection{};
					if (target.Intersect(line.A, line.Direction(), nearIntersection, farIntersection))
					{
						if (!(nearIntersection.distance > 1.f && farIntersection.distance > 1.f))
						{
							nearIntersection.depth = std::clamp(nearIntersection.depth, 0.f, 1.f);
							farIntersection.depth = std::clamp(farIntersection.depth, 0.f, 1.f);

							if (nearIntersection.distance > farIntersection.distance)
							{
								std::swap(nearIntersection, farIntersection);
							}

							float rectified = (farIntersection.distance - nearIntersection.distance) * line.Length();
							local.count++;
							local.overlap += rectified;
							local.total += line.Lerp((nearIntersection.distance + farIntersection.distance) / 2.f);
							local.axis = line.UnitDirection();
						}
					}
				};
			lambda(axes[i], other, myAxisStruct[index]);
			lambda(fumoBox[i], *this, otherAxisStruct[index]);
		}

		rotation_help myPlaceHolder = { .overlap = -INFINITY, .axis = glm::vec3(2, 1, 0), .total = glm::vec3(0.f), .count = 0 };
		rotation_help otherPlaceHolder = { .overlap = -INFINITY, .axis = glm::vec3(2, 1, 0), .total = glm::vec3(0.f), .count = 0 };
		for (std::size_t i = 0; i < 3; i++)
		{
			if (myPlaceHolder.overlap < myAxisStruct[i].overlap)
				myPlaceHolder = myAxisStruct[i];
			if (otherPlaceHolder.overlap < otherAxisStruct[i].overlap)
				otherPlaceHolder = otherAxisStruct[i];
		}
		bool skipRotate = false;
		if (myPlaceHolder.count == 0 && otherPlaceHolder.count == 0)
		{
			skipRotate = true;
		}
		else
		{
			// At least one has non-zero overlap, ensuring this will work
			if (myPlaceHolder.overlap <= otherPlaceHolder.overlap)
				myPlaceHolder = otherPlaceHolder;
			myPlaceHolder.total /= myPlaceHolder.count;

			mostOverlap = myPlaceHolder.overlap;
			rotationAxis = myPlaceHolder.axis;
			rotationPoint = myPlaceHolder.total;
		}

		glm::vec3 oldCenter = this->Center();
		this->matrix[3] = glm::vec4(collide.point, 1);
		if (!skipRotate && std::_Is_finite(mostOverlap) && mostOverlap > EPSILON)
		{
			glm::vec3 lastAxis = glm::normalize(glm::cross(rotationAxis, collide.normal));
			float direction = -(glm::dot(lastAxis, rotationPoint) - glm::dot(lastAxis, this->Center()));
			//direction = -(glm::dot(lastAxis, rotationPoint) - glm::dot(lastAxis, oldCenter));
			if (glm::abs(direction) > EPSILON)
			{
				if (collide.distance > EPSILON)
					this->RotateAbout(glm::rotate(glm::mat4(1.f), collide.distance * glm::sign(direction), rotationAxis), rotationPoint);
			}
		}
	}
	return false;
}

// World is in normalized coordinates so this is trivial
glm::vec3 OrientedBoundingBox::WorldToLocal(const glm::vec3& in) const
{
	return glm::inverse(glm::mat3(this->matrix)) * in;
}

static const std::array<const glm::vec3, 8> multiples = {
	{
		{-1.f, -1.f, -1.f},
		{-1.f, -1.f,  1.f},
		{-1.f,  1.f, -1.f},
		{-1.f,  1.f,  1.f},
		{ 1.f, -1.f, -1.f},
		{ 1.f, -1.f,  1.f},
		{ 1.f,  1.f, -1.f},
		{ 1.f,  1.f,  1.f},
	}
};

static const std::array<std::pair<int, int>, 12> linePairs = {
	{
		{0, 1}, //  0, 0, 1
		{0, 2}, //  0, 1, 0 
		{0, 4}, //  1, 0, 0 
		{2, 3}, //  0, 0, 1 
		{1, 3}, //  0, 1, 0 
		{1, 5}, //  1, 0, 0
		{4, 5}, //  0, 0, 1 
		{4, 6}, //  0, 1, 0 
		{2, 6}, //  1, 0, 0 
		{6, 7}, //  0, 0, 1 
		{5, 7}, //  0, 1, 0 
		{3, 7}  //  1, 0, 0 
	}
};

std::array<LineSegment, 12> OrientedBoundingBox::GetLineSegments() const
{
	std::array<LineSegment, 12> segments{};
	std::array<glm::vec3, 8> points{};
	glm::vec3 center = this->Center();
	points.fill(center);
	for (glm::length_t i = 0; i < 8; i++)
	{
		for (glm::length_t j = 0; j < 3; j++)
		{
			points[i] += (*this)[j] * this->halfs[j] * multiples[i][j];
		}
	}
	for (std::size_t i = 0; i < 12; i++)
	{
		segments[i] = LineSegment(points[linePairs[i].first], points[linePairs[i].second]);
	}
	return segments;
}
