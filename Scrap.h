#pragma once
#ifndef SCRAP_H
#define SCRAP_H
#include <algorithm>
#include "OrientedBoundingBox.h"
#include "StaticOctTree.h"

static OBB smartBox;
static struct
{
	glm::vec3 acceleration{ 0.f };
	glm::vec3 velocity{ 0.f };
	glm::vec3 axisOfGaming{ 0.f };
	OBB* ptr = nullptr;
} smartBoxPhysics;

static float maxRotatePerFrame = 0.f;


// Dumping "smartbox" stuff here because I don't think I'm going to use any of it, but don't wanna outright delete it
// Aligns to the corner
void smartBoxAlignCorner(OBB& other, glm::length_t minDotI, glm::length_t maxDotI)
{
	// Only the scale of the other one is needed to determine if this midpoint is inside
	glm::vec3 dumbScale = other.GetScale();
	glm::vec3 delta = other.Center() - smartBox.Center();

	// Maintain right handedness
	int indexA[3] = { 0, 0, 1 };
	int indexB[3] = { 1, 2, 2 };
	for (int i = 0; i < 3; i++)
	{
		int indexedA = indexA[i];
		int indexedB = indexB[i];

		glm::vec3 axisA = other[indexedA];
		glm::vec3 axisB = other[indexedB];

		// Calculate the distance along each of these axes 
		float projectionA = glm::dot(delta, axisA);
		float projectionB = glm::dot(delta, axisB);

		// See if the extent in that direction is more than covered by these axes
		bool testA = glm::abs(projectionA) >= dumbScale[indexedA];
		bool testB = glm::abs(projectionB) >= dumbScale[indexedB];

		// smartBox collides entirely because of its own sides, therefore it might need to rotate
		if (testA && testB)
		{
			glm::quat current = glm::quat_cast(smartBox.GetNormalMatrix());
			bool axisTest = glm::abs(projectionA) > glm::abs(projectionB);

			// This is the axis of smartBox that will be rotated towards?
			glm::vec3 localAxis = (axisTest) ? axisA : axisB;
			if (glm::sign(glm::dot(localAxis, delta)) > 0)
			{
				localAxis *= -1;
			}
			glm::vec3 rotationAxis = glm::normalize(glm::cross(smartBox[maxDotI], localAxis));

			glm::quat rotation = glm::angleAxis(glm::acos(glm::dot(smartBox[maxDotI], localAxis)), rotationAxis);
			glm::quat newer = glm::normalize(rotation * current);
			glm::quat older = glm::normalize(current);

			float maxDelta = glm::acos(glm::abs(glm::dot(older, newer)));
			float clamped = std::clamp(maxDelta, 0.f, glm::min(1.f, maxRotatePerFrame));
			if (glm::abs(glm::dot(older, newer) - 1) > EPSILON)
			{
				// Slerp interpolates along the shortest axis on the great circle
				smartBox.ReOrient(glm::toMat4(glm::normalize(glm::slerp(older, newer, clamped))));
			}
			break;
		}
	}
}

// Aligns to the Face
void smartBoxAlignFace(OBB& other, glm::vec3 axis, glm::length_t minDotI, glm::length_t maxDotI)
{
	glm::mat3 goobers{ smartBox[0], smartBox[1], smartBox[2] };
	// Leasted aligned keeps its index
	// Middle is replaced with least cross intersection
	// Most is replaced with the negative of new middle cross least
	glm::vec3 least = smartBox[minDotI];                           // goes in smartbox[minDotI]
	glm::vec3 newMost = axis;                              // goes in smartbox[maxDotI]
	glm::vec3 newest = glm::normalize(glm::cross(least, newMost)); // goes in the remaining one(smartbox[3 - minDotI - maxDotI])

	//std::cout << minDotI << ":" << maxDotI << std::endl;

	glm::length_t leastD = minDotI;
	glm::length_t mostD = maxDotI;
	glm::length_t newD = 3 - leastD - mostD;
	if (newD != 3)
	{
		least *= glm::sign(glm::dot(least, goobers[minDotI]));
		newMost *= glm::sign(glm::dot(newMost, goobers[maxDotI]));
		newest *= glm::sign(glm::dot(newest, goobers[newD]));

		glm::mat3 lame{};
		lame[leastD] = least;
		lame[mostD] = newMost;
		lame[newD] = newest;
		glm::quat older = glm::normalize(glm::quat_cast(smartBox.GetNormalMatrix()));
		glm::quat newer = glm::normalize(glm::quat_cast(lame));
		float maxDelta = glm::acos(glm::abs(glm::dot(older, newer)));
		float clamped = std::clamp(maxDelta, 0.f, glm::min(1.f, maxRotatePerFrame));

		// ?
		//if (glm::abs(glm::acos(glm::dot(older, newer))) > EPSILON)
		if (glm::abs(glm::dot(older, newer) - 1) > EPSILON)
		{
			// Slerp interpolates along the shortest axis on the great circle
			smartBox.ReOrient(glm::toMat4(glm::normalize(glm::slerp(older, newer, clamped))));
		}
	}
	else
	{
		std::cout << "Something went horribly wrong" << std::endl;
	}
}

void smartBoxAligner(OBB& other, glm::vec3 axis)
{
	float minDot = INFINITY, maxDot = -INFINITY;
	glm::length_t minDotI = 0, maxDotI = 0;
	for (glm::length_t i = 0; i < 3; i++)
	{
		float local = glm::abs(glm::dot(smartBox[i], axis));
		if (local < minDot)
		{
			minDot = local;
			minDotI = i;
		}
		if (local > maxDot)
		{
			maxDot = local;
			maxDotI = i;
		}
	}
	smartBoxAlignFace(other, axis, minDotI, maxDotI);
}


bool smartBoxCollide()
{
	bool anyCollisions = false;
	smartBoxPhysics.axisOfGaming = glm::vec3{ 0.f };
	smartBoxPhysics.ptr = nullptr;
	StaticOctTree<OBB> staticBoxes;

	auto potentialCollisions = staticBoxes.Search(smartBox.GetAABB());
	int collides = 0;
	//Before(smartBoxPhysics.axisOfGaming);
	float dot = INFINITY;

	for (auto& currentBox : potentialCollisions)
	{
		SlidingCollision c;
		if (smartBox.Overlap(*currentBox, c))
		{
			float temp = glm::dot(c.axis, glm::vec3(0, 1, 0));
			//smartBoxPhysics.velocity += c.axis * c.depth * 0.95f;
			//smartBoxPhysics.velocity -= c.axis * glm::dot(c.axis, smartBoxPhysics.velocity);
			if (temp > 0 && temp <= dot)
			{
				temp = dot;
				smartBoxPhysics.axisOfGaming = c.axis;
				smartBoxPhysics.ptr = &*currentBox;
			}

			float minDot = INFINITY, maxDot = -INFINITY;
			glm::length_t minDotI = 0, maxDotI = 0;
			glm::vec3 upper = c.axis;
			for (glm::length_t i = 0; i < 3; i++)
			{
				float local = glm::abs(glm::dot(smartBox[i], upper));
				if (local < minDot)
				{
					minDot = local;
					minDotI = i;
				}
				if (local > maxDot)
				{
					maxDot = local;
					maxDotI = i;
				}
			}

			smartBoxAlignCorner(*currentBox, minDotI, maxDotI);

			//if (glm::acos(glm::abs(maxDotI - 1)) > EPSILON)
			//if (true) //(c.depth > 0.002) // Why this number
			if (true) // (!(keyState[ArrowKeyRight] || keyState[ArrowKeyLeft]))
			{
				smartBoxAlignFace(*currentBox, upper, minDotI, maxDotI);
				smartBox.OverlapAndSlide(*currentBox);
			}
			else
			{
				smartBox.ApplyCollision(c);
			}
			float oldLength = glm::length(smartBoxPhysics.velocity);
			anyCollisions = true;
		}
	}

	// Scale smart box up a bit to determine axis and
	if (!anyCollisions)
	{
		// This is probably a bad idea

		glm::vec3 oldCenter = smartBox.Center();
		smartBox.Translate(glm::vec3(0, 1, 0) * 2.f * EPSILON);

		potentialCollisions = staticBoxes.Search(smartBox.GetAABB());
		SlidingCollision newest{};
		OBB* newPtr = nullptr;

		for (auto& currentBox : potentialCollisions)
		{
			SlidingCollision c;
			if (smartBox.Overlap(*currentBox, c))
			{
				float temp = glm::dot(c.axis, glm::vec3(0, 1, 0));
				if (temp > 0 && temp <= dot)
				{
					temp = dot;
					smartBoxPhysics.axisOfGaming = c.axis;
					smartBoxPhysics.ptr = &*currentBox;
					newest = c;
					newPtr = &*currentBox;
				}
			}
		}
		smartBox.ReCenter(oldCenter);
		if (newPtr)
		{
			OBB& box = *newPtr;
			float minDot = INFINITY, maxDot = -INFINITY;
			glm::length_t minDotI = 0, maxDotI = 0;
			glm::vec3 upper = newest.axis;
			for (glm::length_t i = 0; i < 3; i++)
			{
				float local = glm::abs(glm::dot(smartBox[i], upper));
				if (local < minDot)
				{
					minDot = local;
					minDotI = i;
				}
				if (local > maxDot)
				{
					maxDot = local;
					maxDotI = i;
				}
			}
			smartBoxAlignCorner(box, minDotI, maxDotI);
			smartBoxAlignFace(box, upper, minDotI, maxDotI);
		}
	}

	return anyCollisions;
}


#endif // SCRAP_H