#include "StaticOctTree.h"

 StaticOctTree::StaticOctTree(const glm::vec3& negativeBound, const glm::vec3& positiveBound) : bounds(negativeBound, positiveBound)
{

}

StaticOctTree::~StaticOctTree()
{
	if (this->tree != nullptr)
	{
		for (int i = 0; i < 8; i++)
		{
			if (this->tree[i] != nullptr)
			{
				delete this->tree[i];
			}
		}
		delete[] this->tree;
	}
}