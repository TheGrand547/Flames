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
	AABB local(this->halfs * 2.f); // Why is this multiplied by 2?
	glm::vec3 transformed = this->WorldToLocal(other.center - this->Center());
	Sphere temp{ other.radius, transformed };
	bool result = local.Overlap(temp, collision);
	collision.normal = this->matrix * glm::vec4(collision.normal, 0);
	collision.point  = other.center + collision.normal * collision.depth;
	return result;
}

bool OrientedBoundingBox::Overlap(const OrientedBoundingBox& other, SlidingCollision& slide, RotationCollision& rotate) const
{
	rotate.Clear();
	if (this->Overlap(other, slide))
	{
		std::array<LineSegment, 12> axes = this->GetLineSegments(), fumoBox = other.GetLineSegments();

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
					RayCollision nearIntersection{}, farIntersection{};
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

		struct rotation_help_sorter {
			bool operator()(const rotation_help& lhs, const rotation_help& rhs)
			{
				return lhs.overlap < rhs.overlap;
			}
		};
		std::sort(myAxisStruct.begin(), myAxisStruct.end(), rotation_help_sorter());
		std::sort(otherAxisStruct.begin(), otherAxisStruct.end(), rotation_help_sorter());
		myPlaceHolder = myAxisStruct[2];
		otherPlaceHolder = otherAxisStruct[2];
		bool skipRotate = false;
		// If two sides have equal overlap or both fully overlap this is an invalid collision
		if ((myAxisStruct[2].overlap > 1.f && myAxisStruct[1].overlap > 1.f) ||
			(otherAxisStruct[2].overlap > 1.f && otherAxisStruct[1].overlap > 1.f) ||
			myAxisStruct[2].overlap == myAxisStruct[1].overlap ||
			otherAxisStruct[2].overlap == otherAxisStruct[1].overlap)
		{
			skipRotate = true;
		}
		if (myPlaceHolder.count == 0 && otherPlaceHolder.count == 0)
		{
			skipRotate = true;
		}
		else
		{
			// At least one has non-zero overlap, ensuring this will work
			if (myPlaceHolder.overlap <= otherPlaceHolder.overlap || myPlaceHolder.count == 0)
				myPlaceHolder = otherPlaceHolder;

			myPlaceHolder.total /= myPlaceHolder.count;

			rotate.axis = myPlaceHolder.axis;
			rotate.point = myPlaceHolder.total;
		}

		glm::vec3 oldCenter = this->Center();
		if (!skipRotate && std::_Is_finite(myPlaceHolder.overlap) && myPlaceHolder.overlap > EPSILON)
		{
			glm::vec3 mostAlignedVector(0.f), mostAlignedVector2(0.f);
			float mostAlignedDot = -INFINITY, mostAlignedDot2(0.f);
			for (glm::length_t i = 0; i < 3; i++)
			{
				float local = glm::abs(glm::dot((*this)[i], slide.normal));
				float local2 = glm::abs(glm::dot(other[i], slide.normal));
				if (local > mostAlignedDot)
				{
					mostAlignedDot = local;
					mostAlignedVector = (*this)[i];
				}
				if (local2 > mostAlignedDot2)
				{
					mostAlignedDot2 = local2;
					mostAlignedVector2 = other[i];
				}
			}

			glm::vec3 lastAxis = glm::normalize(glm::cross(rotate.axis, slide.normal));
			float direction = -(glm::dot(lastAxis, rotate.point) - glm::dot(lastAxis, this->Center()));
			//direction = -(glm::dot(lastAxis, rotationPoint) - glm::dot(lastAxis, oldCenter));
			if (glm::abs(direction) > EPSILON)
			{
				//float maximum = glm::acos(1 - (glm::abs(glm::dot(mostAlignedVector, rotationAxis))));
				float maximum = glm::acos(glm::abs(glm::dot(mostAlignedVector, rotate.axis)));
				//collide.distance = glm::min(maximum, collide.distance);
				//if (collide.distance > EPSILON)
				rotate.distance = slide.distance * glm::sign(direction);
			}
			else
			{
				rotate.distance = 0.f;
			}
		}
		return true;
	}
	return false;
}

bool OrientedBoundingBox::OverlapCompleteResponse(const OrientedBoundingBox& other)
{
	
	SlidingCollision slide{};
	RotationCollision rotate{};
	bool result = this->Overlap(other, slide, rotate);
	if (result)
	{
		this->ApplyCollision(slide);
		std::cout << "Rotation axis: " << rotate.axis << std::endl;
		this->ApplyCollision(rotate);
	}
	return result;
}

// World is in normalized coordinates so this is trivial
glm::vec3 OrientedBoundingBox::WorldToLocal(const glm::vec3& in) const
{
	// Inverse of an ortho-normal matrix is it's transpose
	//return glm::inverse(glm::mat3(this->matrix)) * in;
	return glm::transpose(glm::mat3(this->matrix)) * in;
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

// What in gods name is this comment thing
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

std::vector<Triangle> OrientedBoundingBox::GetTriangles() const
{
	std::vector<Triangle> triangles;
	if (!glm::all(glm::equal(this->halfs, glm::vec3(0))))
	{
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
		if (this->halfs[1] != 0.f && this->halfs[2] != 0.f) // Forward/Backward planes needed
		{
			// 4,5,6,7 (+x)
			triangles.emplace_back(points[4], points[6], points[5]);
			triangles.emplace_back(points[5], points[6], points[7]);
			// 0,1,2,3 (-x)
			triangles.emplace_back(points[2], points[0], points[1]);
			triangles.emplace_back(points[2], points[1], points[3]);
		}
		if (this->halfs[0] != 0.f && this->halfs[2] != 0.f) // Up/Down planes needed
		{
			// 2,3,6,7 (+y)
			triangles.emplace_back(points[2], points[3], points[6]);
			triangles.emplace_back(points[6], points[3], points[7]);
			// 0,1,4,5 (-y)
			triangles.emplace_back(points[1], points[0], points[4]);
			triangles.emplace_back(points[1], points[4], points[5]);
		}
		if (this->halfs[0] != 0.f && this->halfs[1] != 0.f) // Left/Right planes needed
		{
			// 1,3,5,7 (+z)
			triangles.emplace_back(points[3], points[1], points[5]);
			triangles.emplace_back(points[3], points[5], points[7]);
			// 0,2,4,6 (-z)
			triangles.emplace_back(points[0], points[2], points[4]);
			triangles.emplace_back(points[4], points[2], points[6]);
		}
	}
	return triangles;
}
