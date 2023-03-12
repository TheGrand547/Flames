#pragma once
#ifndef STATIC_OCT_TREE_H
#define STATIC_OCT_TREE_H
#include <glm/glm.hpp>
#include <vector>
#include "AABC.h"

class StaticOctTree
{
protected:
	StaticOctTree** tree = nullptr;
	std::vector<void*> pointers;
	AABC bounds;
public:
	StaticOctTree(const glm::vec3& negativeBound, const glm::vec3& positiveBound);
	~StaticOctTree();

};

#endif // STATIC_OCT_TREE_H
