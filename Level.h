#pragma once
#ifndef LEVEL_H
#define LEVEL_H
#include "kdTree.h"
#include "StaticOctTree.h"
#include "DynamicTree.h"
#include "PathNode.h"

namespace Level
{
	static StaticOctTree<OBB> Geometry;
	static kdTree<PathNodePtr> Tree;
	static std::vector<PathNodePtr> AllNodes;

	void Clear() noexcept;
}


#endif // LEVEL_H