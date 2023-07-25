#include "StaticOctTree.h"

#define OCT 8

StaticOctTree::StaticOctTree(const glm::vec3& negativeBound, const glm::vec3& positiveBound, int depth) : bounds(negativeBound, positiveBound), depth(depth + 1), 
							leaf(this->depth > MAX_OCT_TREE_DEPTH || this->bounds.Volume() < MIN_OCT_TREE_VOLUME)
{

}

StaticOctTree::StaticOctTree(const AABB& bounds, int depth) : bounds(bounds), depth(depth + 1), leaf(this->depth > MAX_OCT_TREE_DEPTH || this->bounds.Volume() < MIN_OCT_TREE_VOLUME)
{

}


StaticOctTree::StaticOctTree(const glm::vec3& negativeBound, const glm::vec3& positiveBound) : bounds(negativeBound, positiveBound), depth(0), leaf(false)
{

}

StaticOctTree::~StaticOctTree()
{
	this->Clear();
}

void StaticOctTree::Clear()
{
	if (this->tree != nullptr)
	{
		for (int i = 0; i < 8; i++)
		{
			if (this->tree[i] != nullptr)
			{
				this->tree[i]->Clear();
				delete this->tree[i];
			}
		}
		delete[] this->tree;
		this->tree = nullptr;
	}
	this->pointers.clear();
}


void StaticOctTree::Generate()
{
	if (!this->leaf)
	{
		this->Clear();
		this->tree = new StaticOctTree * [OCT];
		if (this->tree)
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
				AABB current = AABB::MakeAABB(center, center + extents * mults);
				mults.x *= -1.f;
				if (i % 2 == 1)
				{
					mults.y *= -1.f;
				}
				if (i == 3)
				{
					mults.z *= -1.f;
				}

				this->tree[i] = new StaticOctTree(current, this->depth);
				if (!this->tree[i])
				{
					// TODO: LOG PANIC
					this->Clear();
					return;
				}
			}
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
	if (this->leaf)
	{
		this->pointers.push_back(element);
	}
	else
	{
		unsigned char indicator = 0x00;
		if (!this->tree)
			this->Generate();
		if (!this->tree)
		{
			// TODO: LOG ERROR
			return;
		}
		for (std::size_t i = 0; i < 8; i++)
		{
			indicator |= this->tree[i]->bounds.Overlap(box) << i;

		}
		if (std::has_single_bit(indicator))
		{
			this->tree[std::bit_width(indicator) - 1]->InsertQuick(element, box);
		}
		else
		{
			this->pointers.push_back(element);
		}
	}
}
