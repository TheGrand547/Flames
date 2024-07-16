#pragma once
#ifndef STATIC_OCT_TREE_H
#define STATIC_OCT_TREE_H
#include <glm/glm.hpp>
#include <array>
#include <list>
#include <vector>
#include "AABB.h"
#include "OrientedBoundingBox.h"

#ifndef OCT_TREE_MAX_DEPTH
#define OCT_TREE_MAX_DEPTH (5)
#endif // OCT_TREE_MAX_DEPTH

#ifndef OCT_TREE_MIN_VOLUME
#define OCT_TREE_MIN_VOLUME (10.f)
#endif // OCT_TREE_MIN_VOLUME

#ifndef OCT_TREE_DIMENSION
#define OCT_TREE_DIMENSION (100.f)
#endif // OCT_TREE_DIMENSION

// HEAVILY inspired by olc's oct tree video and implementation
template<class T>
class StaticOctTree
{
	using Item = typename std::list<T>::iterator;
protected:
	class StaticOctTreeNode
	{
	protected:
		std::array<std::unique_ptr<StaticOctTreeNode>, 8> tree;
		std::array<AABB, 8> internals;

		std::vector<std::pair<AABB, Item>> objects;
		const AABB bounds;
		const int depth;

		void Generate();
	public:
		StaticOctTreeNode(const glm::vec3& negativeBound = glm::vec3(-OCT_TREE_DIMENSION),
			const glm::vec3& positiveBound = glm::vec3(OCT_TREE_DIMENSION), int depth = 0);
		StaticOctTreeNode(const AABB& bound, int depth = 0);
		~StaticOctTreeNode();

		void Clear();

		std::list<Item> Dump() const;
		void Dump(std::list<Item>& items) const;

		std::list<Item> Search(const AABB& box) const;
		void Search(const AABB& box, std::list<Item>& items) const;
		void Insert(const Item& element, const AABB& box);

		std::list<Item> RayCast(const Ray& line) const;
		void RayCast(const Ray& line, std::list<Item>& items) const;
	};

	StaticOctTreeNode root;
	std::list<T> items;
public:
	StaticOctTree();
	StaticOctTree(const glm::vec3& bounds);
	StaticOctTree(const AABB& bound);
	~StaticOctTree();

	constexpr bool empty() const { return this->items.empty(); }

	typename std::list<T>::iterator begin() { return this->items.begin(); }
	typename std::list<T>::const_iterator cbegin() const { return this->items.cbegin(); }
	typename std::list<T>::iterator end() { return this->items.end(); }
	typename std::list<T>::const_iterator cend() const { return this->items.cend(); }

	std::list<Item> Search(const AABB& area) const;
	std::list<Item> DepthSearch(const AABB& area);
	std::list<Item> RayCast(const Ray& line) const;
	std::list<Item> RayCastSort(const Ray& line) const;
	T* RayCastFirst(const Ray& ray) const;

	constexpr std::size_t size() const { return this->items.size(); }

	void Clear();
	void Insert(const T& element, const AABB& box);
};

template<class T> using Item = typename std::list<T>::iterator;

template<class T>
inline StaticOctTree<T>::StaticOctTree() : root()
{
}

template<class T>
inline StaticOctTree<T>::StaticOctTree(const glm::vec3& bounds) : root(-glm::abs(bounds), glm::abs(bounds))
{
}

template<class T>
inline StaticOctTree<T>::StaticOctTree(const AABB& bounds) : root(bounds)
{
}

template<class T>
inline StaticOctTree<T>::~StaticOctTree()
{
	this->Clear();
}

template<class T>
inline std::list<typename std::list<T>::iterator> StaticOctTree<T>::Search(const AABB& area) const
{
	return this->root.Search(area);
}

template<class T>
inline std::list<typename std::list<T>::iterator> StaticOctTree<T>::DepthSearch(const AABB& area)
{
	std::list<typename std::list<T>::iterator> temp = this->root.Search(area);
	// TODO: Depth based sort
	return temp;
}

template<class T>
std::list<typename std::list<T>::iterator> StaticOctTree<T>::RayCast(const Ray& line) const
{
	return this->root.RayCast(line);
}

template<class T>
std::list<typename std::list<T>::iterator> StaticOctTree<T>::RayCastSort(const Ray& line) const
{
	Log("TODO IMPLEMENT DUMMY");
	return std::list<Item>();
}

/*
template<class T>
std::list<typename std::list<T>::iterator> StaticOctTree<T>::RayCastSort(const Ray& line) const
{
	struct RaySort
	{
		Item item;
		float distance;
	};
	std::list<Item> items = this->root.RayCast(line);
	// This *really* needs a depth sort thing
	return this->root.RayCast(line);
}*/

template<class T>
T* StaticOctTree<T>::RayCastFirst(const Ray& line) const
{
	std::list<Item> hits = this->root.RayCast(line);
	if (hits.size())
	{
		float furthest = INFINITY;
		T* closestPtr = hits[0];
		RayCollision collision{};
		for (Item& item : hits)
		{

		}
		return closestPtr;
	}
	return nullptr;
}

template<class T>
inline void StaticOctTree<T>::Clear()
{
	this->items.clear();
	this->root.Clear();
}

template<class T>
inline void StaticOctTree<T>::Insert(const T& element, const AABB& box)
{
	this->items.push_back(element);
	this->root.Insert(std::prev(this->items.end()), box);
}


// The OctTreeNodes
template<class T>
inline StaticOctTree<T>::StaticOctTreeNode::StaticOctTreeNode(const glm::vec3& negativeBound, const glm::vec3& positiveBound, int depth)
	: bounds(negativeBound, positiveBound), depth(depth)
{
	this->Generate();
}

template<class T>
inline StaticOctTree<T>::StaticOctTreeNode::StaticOctTreeNode(const AABB& bounds, int depth) : bounds(bounds), depth(depth)
{
	this->Generate();
}

template<class T>
inline StaticOctTree<T>::StaticOctTreeNode::~StaticOctTreeNode()
{
	this->Clear();
}

template<class T>
inline void StaticOctTree<T>::StaticOctTreeNode::Generate()
{
	// Layout of the tree, in terms of the half lengths added to the center, 
	// -x, -y, -z Index 0
	//  x, -y, -z Index 1
	// -x,  y, -z Index 2
	//  x,  y, -z Index 3
	// -x, -y,  z Index 4
	//  x, -y,  z Index 5	
	// -x,  y,  z Index 6
	//  x,  y,  z Index 7
	glm::vec3 center = this->bounds.GetCenter();
	glm::vec3 extents = this->bounds.Deviation();

	glm::vec3 mults(-1.f);
	for (std::size_t i = 0; i < 8; i++)
	{
		this->internals[i] = AABB::MakeAABB(center, center + extents * mults);
		mults.x *= -1.f;
		if (i % 2 == 1)
		{
			mults.y *= -1.f;
		}
		if (i == 3)
		{
			mults.z *= -1.f;
		}
	}
}

template<class T>
inline void StaticOctTree<T>::StaticOctTreeNode::Clear()
{
	this->objects.clear();
	for (std::size_t i = 0; i < 8; i++)
	{
		this->tree[i].reset();
	}
}

template<class T>
inline std::list<Item<typename T>> StaticOctTree<T>::StaticOctTreeNode::Dump() const
{
	std::list<Item> result();
	this->Dump(result);
	return result;
}

template<class T>
inline void StaticOctTree<T>::StaticOctTreeNode::Dump(std::list<Item>& list) const
{
	std::copy(this->objects.front(), this->objects.back(), std::back_inserter(list));
	for (std::size_t i = 0; i < 8; i++)
	{
		if (this->tree[i])
		{
			this->tree[i]->Dump(list);
		}
	}
}

template<class T>
inline std::list<Item<typename T>> StaticOctTree<T>::StaticOctTreeNode::Search(const AABB& box) const
{
	std::list<typename std::list<T>::iterator> items;
	this->Search(box, items);
	return items;
}

template<class T>
inline void StaticOctTree<T>::StaticOctTreeNode::Search(const AABB& box, std::list<Item>& items) const
{
	for (const auto& element : this->objects)
	{
		if (box.Overlap(element.first))
			items.push_back(element.second);
	}
	for (std::size_t i = 0; i < 8; i++)
	{
		if (this->tree[i])
		{
			if (box.Contains(this->internals[i]))
			{
				this->tree[i]->Dump(items);
			}
			else if (this->internals[i].Overlap(box))
			{
				this->tree[i]->Search(box, items);
			}
		}
	}
}

template<class T>
inline void StaticOctTree<T>::StaticOctTreeNode::Insert(const Item& element, const AABB& box)
{
	for (std::size_t i = 0; i < 8; i++)
	{
		glm::vec3 center = this->internals[i].GetCenter();
		glm::vec3 deviation = this->internals[i].Deviation();
		if (this->internals[i].Contains(box))
		{
			if (this->depth + 1 < OCT_TREE_MAX_DEPTH)
			{
				if (!this->tree[i])
				{
					this->tree[i] = std::make_unique<StaticOctTreeNode>(this->internals[i], depth + 1);
				}
				this->tree[i]->Insert(element, box);
				return;
			}
		}
	}
	this->objects.push_back({ box, element });
}

template<class T>
std::list<Item<typename T>> StaticOctTree<T>::StaticOctTreeNode::RayCast(const Ray& line) const
{
	std::list<Item> items{};
	this->RayCast(line, items);
	return items;
}

template<class T>
void StaticOctTree<T>::StaticOctTreeNode::RayCast(const Ray& line, std::list<Item>& items) const
{
	if (this->bounds.FastIntersect(line.initial, line.direction))
	{
		for (auto& element : this->objects)
		{
			if (element.first.FastIntersect(line.initial, line.direction))
			{
				items.push_back(element.second);
			}
		}
		for (auto& branch : this->tree)
		{
			if (branch)
			{
				branch->RayCast(line, items);
			}
		}
	}
}

#endif // STATIC_OCT_TREE_H
