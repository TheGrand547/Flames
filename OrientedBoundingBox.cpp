#include "OrientedBoundingBox.h"
#include <algorithm>
#include <bit>

OrientedBoundingBox::OrientedBoundingBox(const glm::vec3& euler, const glm::vec3& deltas) noexcept : matrix(1.f), halfs(glm::abs(deltas))
{
	this->Rotate(euler);
}

OrientedBoundingBox::OrientedBoundingBox(const Model& model) noexcept : OrientedBoundingBox(model.rotation, model.scale)
{
	this->matrix[3] = glm::vec4(model.translation, 1);
}

bool OrientedBoundingBox::Intersection(const Plane& plane, Collision& collision) const noexcept
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

bool OrientedBoundingBox::FastIntersect(const glm::vec3& point, const glm::vec3& dir) const noexcept
{
	glm::vec3 delta = glm::vec3(this->Center()) - point;
	float nearHit = -std::numeric_limits<float>::infinity(), farHit = std::numeric_limits<float>::infinity();
	for (auto i = 0; i < 3; i++)
	{
		glm::vec3 axis = this->matrix[i];
		float scale = this->halfs[i];
		float parallel = glm::dot(axis, delta);
		if (glm::abs(glm::dot(dir, axis)) < EPSILON)
		{
			if (-parallel - scale > 0 || -parallel + scale > 0)
			{
				return false;
			}
		}

		float scaling = glm::dot(axis, dir);
		float param0 = (parallel + scale) / scaling;
		float param1 = (parallel - scale) / scaling;

		if (param0 > param1)
		{
			std::swap(param0, param1);
		}
		if (param0 > nearHit)
		{
			nearHit = param0;
		}
		if (param1 < farHit)
		{
			farHit = param1;
		}
		if (nearHit > farHit)
		{
			return false;
		}
		if (farHit < 0)
		{
			return false;
		}
	}
	return true;
}

// https://www.sciencedirect.com/topics/computer-science/oriented-bounding-box
bool OrientedBoundingBox::Intersect(const glm::vec3& point, const glm::vec3& dir, RayCollision& nearHit, RayCollision& farHit) const noexcept
{
	// TODO: For line segments do the clamp thingy
	nearHit.Clear();
	farHit.Clear();
	nearHit.distance = -std::numeric_limits<float>::infinity();
	farHit.distance = std::numeric_limits<float>::infinity();

	glm::vec3 delta = this->Center() - point;
	for (auto i = 0; i < 3; i++)
	{
		glm::vec3 axis = this->matrix[i];
		float scale = this->halfs[i];

		float parallel = glm::dot(axis, delta); // Distance from Point to my center, in the direction of this axis
		float scaling = glm::dot(axis, dir);    // Length of projection of dir onto this axis

		// Check if the direction is parallel to one of the faces
		if (glm::abs(scaling) < EPSILON)
		{
			if (abs(parallel) > scale)
			{
				return false;
			}
			else
			{
				//continue;
			}
		}

		float param0 = (parallel + scale) / scaling;
		float param1 = (parallel - scale) / scaling;

		if (param0 > param1)
		{
			std::swap(param0, param1);
		}
		if (param0 > nearHit.distance)
		{
			nearHit.distance = param0;
			nearHit.normal = axis * glm::sign(-parallel);
		}
		if (param1 < farHit.distance)
		{
			farHit.distance = param1;
			farHit.normal = axis * glm::sign(parallel);
		}
		if (nearHit.distance > farHit.distance)
		{
			return false;
		}
		if (farHit.distance < 0)
		{
			return false;
		}
	}
	nearHit.point = nearHit.distance * dir + point;
	farHit.point = farHit.distance * dir + point;
	if (nearHit.distance < 0)
	{
		std::swap(nearHit, farHit);
	}
	return true;
}

bool OrientedBoundingBox::Overlap(const Capsule& other, Collision& collide) const noexcept
{
	return this->Overlap(Sphere(other.GetRadius(), other.ClosestPoint(this->Center())), collide);
}

bool OrientedBoundingBox::Overlap(const Sphere& other, Collision& collision) const noexcept
{
	AABB local(this->halfs * 2.f); // Why is this multiplied by 2?
	glm::vec3 transformed = this->WorldToLocal(other.center);
	Sphere temp{ other.radius, transformed };
	bool result = local.Overlap(temp, collision);
	collision.normal = this->matrix * glm::vec4(collision.normal, 0);
	collision.point  = other.center + collision.normal * collision.depth;
	return result;
}

// https://web.stanford.edu/class/cs273/refs/obb.pdf
bool OrientedBoundingBox::Overlap(const OrientedBoundingBox& other, SlidingCollision& result) const noexcept
{
	const glm::mat<3, 2, const glm::length_t> indexLookUp{ {2, 1}, {2, 0}, { 1, 0} };
	std::array<glm::vec3, 15> separatingAxes{};
	glm::mat3 dotProducts{};

	if (!std::is_constant_evaluated())
	{
		dotProducts = glm::transpose(glm::mat3(other.matrix)) * glm::mat3(this->matrix);
		for (glm::length_t i = 0; i < 3; i++)
		{
			dotProducts[i] = glm::abs(dotProducts[i]);
#ifdef _DEBUG
			for (glm::length_t j = 0; j < 3; j++)
			{
				assert(dotProducts[i][j] == glm::abs(glm::dot(glm::vec3(this->matrix[i]), other[j])));
			}
#endif // _DEBUG
		}
	}
	else
	{
		for (glm::length_t i = 0; i < 3; i++)
		{
			glm::vec3 myAxis = this->matrix[i];
			for (glm::length_t j = 0; j < 3; j++)
			{
				dotProducts[i][j] = glm::abs(glm::dot(myAxis, other[j]));
			}
		}
	}

	// If the least separating axis is one of the 6 face normals it's a corner-edge collision, otherwise it's edge-edge
	const glm::vec3 delta = this->Center() - other.Center();
	result.distance = INFINITY;
	glm::length_t index = 0;
	for (glm::length_t i = 0; i < separatingAxes.size(); i++)
	{
		float deltaProjection = 0, obbProjections = 0;
		glm::length_t truncatedIndex = i % 5;
		glm::length_t axialIndex = i / 5;
		if (truncatedIndex == 0) // Axis from this OBB
		{
			// Works because the dot products to be multipled form a column in the dot product matrix
			obbProjections = this->halfs[axialIndex] + glm::dot(other.halfs, dotProducts[axialIndex]);
			separatingAxes[i] = (*this)[axialIndex];
		}
		else if (truncatedIndex == 1) // Axis from the other OBB
		{
			obbProjections = other.halfs[axialIndex] + this->halfs[0] * dotProducts[0][axialIndex] + this->halfs[1] * dotProducts[1][axialIndex]
				+ this->halfs[2] * dotProducts[2][axialIndex];
			separatingAxes[i] = other[axialIndex];
		}
		else
		{
			// Truncated index is [2,4], map to [0, 2]
			// Axis is this[axialIndex] cross other[truncatedIndex - 2]
			truncatedIndex -= 2;

			glm::vec3 myAxis = this->matrix[axialIndex];
			glm::vec3 otherAxis = other.matrix[truncatedIndex];
			glm::vec3 crossResult = glm::cross(myAxis, glm::vec3(otherAxis));

			float crossLength = 1.f / glm::length(crossResult);
			separatingAxes[i] = crossResult * crossLength;


			glm::length_t firstPairA = indexLookUp[axialIndex][0], firstPairB = indexLookUp[axialIndex][1];
			glm::length_t secondPairA = indexLookUp[truncatedIndex][0], secondPairB = indexLookUp[truncatedIndex][1];


			obbProjections += this->halfs[firstPairA] * dotProducts[firstPairB][truncatedIndex] +
				this->halfs[firstPairB] * dotProducts[firstPairA][truncatedIndex];

			obbProjections += other.halfs[secondPairA] * dotProducts[axialIndex][secondPairB] +
				other.halfs[secondPairB] * dotProducts[axialIndex][secondPairA];
			obbProjections *= crossLength;
			truncatedIndex += 2;
		}
		deltaProjection = glm::abs(glm::dot(separatingAxes[i], delta));

		// In Case of Collision Whackiness break glass
#ifdef _DEBUG
		float cardinal = 0.f;
		glm::vec3 axis = separatingAxes[i];
		for (glm::length_t i = 0; i < 3; i++)
		{
			cardinal += glm::abs(this->halfs[i] * glm::dot(glm::vec3(this->matrix[i]), axis));
			cardinal += glm::abs(other.halfs[i] * glm::dot(glm::vec3(other.matrix[i]), axis));
		}

		if (glm::abs(obbProjections - cardinal) > EPSILON)
		{
			std::cout << "Error in OBB Optimization: " << i << ":" << axialIndex <<
				":" << truncatedIndex << ":" << axis << ":" << obbProjections << "\t" << cardinal << std::endl;
		}
#endif // _DEBUG
		const float overlap = obbProjections - deltaProjection;
		// This axis is a separating one 
		if (deltaProjection > obbProjections)
		{
			return false;
		}
		// Find the minimum axis projection
		if (result.distance > overlap)
		{
			index = i;
			result.distance = overlap;
		}
	}
	// Annoying warning from implying index might not satisfy 0 <= index <= 15 - 1, when it can only be one of them
	// Result.normal is the direction this OBB needs to head to escape collision

#pragma warning( suppress : 28020 )
	result.normal = separatingAxes[index] * glm::sign(glm::dot(delta, separatingAxes[index]));
	result.point = this->Center() + result.distance * result.normal;
	return true;
}


// TODO: Remove
bool OrientedBoundingBox::Overlap(const OrientedBoundingBox& other, SlidingCollision& slide, RotationCollision& rotate) const noexcept
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

bool OrientedBoundingBox::OverlapCompleteResponse(const OrientedBoundingBox& other) noexcept
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
glm::vec3 OrientedBoundingBox::WorldToLocal(const glm::vec3& in) const noexcept
{
	// Inverse of an ortho-normal matrix is its transpose
	// (3x3)T * 3x1 == 1x3 * 3x3 in this situation, due to how glm does math
	return (in - this->Center()) * glm::mat3(this->matrix);
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

std::array<LineSegment, 12> OrientedBoundingBox::GetLineSegments() const noexcept
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

std::vector<Triangle> OrientedBoundingBox::GetTriangles() const noexcept
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
