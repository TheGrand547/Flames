#pragma once
#ifndef STATIC_OCT_TREE_H
#define STATIC_OCT_TREE_H
#include <glm/glm.hpp>
#include <vector>
#include "AABB.h"
#include "OrientedBoundingBox.h"

#ifndef MAX_OCT_TREE_DEPTH
#define MAX_OCT_TREE_DEPTH (5.f)
#endif // MAX_OCT_TREE_DEPTH

#ifndef MIN_OCT_TREE_VOLUME
#define MIN_OCT_TREE_VOLUME (10.f)
#endif // MIN_OCT_TREE_VOLUME

class StaticOctTree
{
protected:
	StaticOctTree** tree = nullptr;
	std::vector<OBB> pointers; // TODO: make this not shit
	const AABB bounds;
	const int depth;
	const bool leaf;

	StaticOctTree(const glm::vec3& negativeBound, const glm::vec3& positiveBound, int depth);
	StaticOctTree(const AABB& bounds, int depth);

	bool CollideQuick(const OBB& element, const AABB& box) const;

	void InsertQuick(const OBB& element, const AABB& box);
public:
	 StaticOctTree(const glm::vec3& negativeBound = glm::vec3(-100, -100, -100), const glm::vec3& positiveBound = glm::vec3(100, 100, 100));
	~StaticOctTree();

	bool Collide(const OBB& element) const;

	void Clear();
	void Generate();

	void Insert(const OBB& element);
};

#endif // STATIC_OCT_TREE_H
