#include "StaticOctTree.h"

constexpr auto OCT = 8;

StaticOctTree::StaticOctTree(const glm::vec3& negativeBound, const glm::vec3& positiveBound, int depth) : bounds(negativeBound, positiveBound), depth(depth + 1)
{
	this->Generate();
}


StaticOctTree::StaticOctTree(const AABB& bound) : bounds(bound), depth(0)
{
	this->Generate();
}

StaticOctTree::StaticOctTree(const AABB& bounds, int depth) : bounds(bounds), depth(depth + 1)
{
	this->Generate();
}


StaticOctTree::StaticOctTree(const glm::vec3& negativeBound, const glm::vec3& positiveBound) : bounds(negativeBound, positiveBound), depth(0)
{
	this->Generate();
}

StaticOctTree::~StaticOctTree()
{
	this->Clear();
}

bool StaticOctTree::CollideQuick(const OBB& element, const AABB& box) const
{
	if (!box.Overlap(this->bounds))
		return false;
	for (const OBB& guy : this->pointers)
	{
		if (guy.Overlap(box) && guy.Overlap(element))
			return true;
	}
	/*
	if (!this->leaf)
		for (std::size_t i = 0; i < OCT; i++)
			if (this->tree[i] && this->tree[i]->CollideQuick(element, box))
				return true;
				*/
	return false;
}

bool StaticOctTree::Collide(const OBB& element) const
{
	AABB box = element.GetAABB();
	return this->CollideQuick(element, box);
}

void StaticOctTree::Clear()
{
	for (int i = 0; i < OCT; i++)
	{
		if (this->tree[i] != nullptr)
		{
			this->tree[i]->Clear();
			delete this->tree[i];
		}
	}
	this->pointers.clear();
}


void StaticOctTree::Generate()
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
	for (std::size_t i = 0; i < OCT; i++)
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

void StaticOctTree::Insert(const OBB& element)
{
	AABB box = element.GetAABB();
	if (!box.Overlap(this->bounds))
		return;
	this->InsertQuick(element, box);
}

void StaticOctTree::InsertQuick(const OBB& element, const AABB& box)
{
	if (this->depth + 1 > MAX_OCT_TREE_DEPTH)
	{
		this->pointers.push_back(element);
	}
	else
	{
		for (std::size_t i = 0; i < OCT; i++)
		{
			if (this->internals[i].Contains(box))
			{
				if (!this->tree[i])
				{
					this->tree[i] = new StaticOctTree(this->internals[i], this->depth + 1);
				}
				this->tree[i]->InsertQuick(element, box);
				return;
			}
		}
		this->pointers.push_back(element);
	}
}
