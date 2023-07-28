#pragma once
#ifndef STATIC_OCT_TREE_H
#define STATIC_OCT_TREE_H
#include <glm/glm.hpp>
#include <array>
#include <vector>
#include "AABB.h"
#include "OrientedBoundingBox.h"

#ifndef MAX_OCT_TREE_DEPTH
#define MAX_OCT_TREE_DEPTH (5)
#endif // MAX_OCT_TREE_DEPTH

#ifndef MIN_OCT_TREE_VOLUME
#define MIN_OCT_TREE_VOLUME (10.f)
#endif // MIN_OCT_TREE_VOLUME

#ifndef DEFAULT_OCT_TREE_DIMENSION
#define DEFAULT_OCT_TREE_DIMENSION (100.f)
#endif // DEFAULT_OCT_TREE_DIMENSION

// TODO: don't just have this based on an OBB
class StaticOctTree
{
protected:
	std::array<StaticOctTree*, 8> tree;
	std::array<AABB, 8> internals;
	std::vector<OBB> pointers; // TODO: make this not shit
	const AABB bounds;
	const int depth;

	StaticOctTree(const glm::vec3& negativeBound, const glm::vec3& positiveBound, int depth);
	StaticOctTree(const AABB& bounds, int depth);

	bool CollideQuick(const OBB& element, const AABB& box) const;

	void Generate();
	void InsertQuick(const OBB& element, const AABB& box);
public:
	StaticOctTree(const glm::vec3& negativeBound = glm::vec3(-DEFAULT_OCT_TREE_DIMENSION), const glm::vec3& positiveBound = glm::vec3(DEFAULT_OCT_TREE_DIMENSION));
	StaticOctTree(const AABB& bound); 
	~StaticOctTree();

	bool Collide(const OBB& element) const;

	void Clear();

	void Insert(const OBB& element);
};

#endif // STATIC_OCT_TREE_H
