#pragma once
#ifndef DYNAMIC_OCT_TREE_H
#define DYNAMIC_OCT_TREE_H
#include <list>
#include "AABB.h"

#ifndef DYNAMIC_OCT_TREE_MAX_DEPTH
#define DYNAMIC_OCT_TREE_MAX_DEPTH (5)
#endif //DYNAMIC_OCT_TREE_MAX_DEPTH

#ifndef DYNAMIC_OCT_TREE_DIMENSION
#define DYNAMIC_OCT_TREE_DIMENSION (100.f)
#endif // DYNAMIC_OCT_TREE_DIMENSION

#ifndef DYNAMIC_OCT_TREE_MIN_VOLUME
#define DYNAMIC_OCT_TREE_MIN_VOLUME (10.f)
#endif // DYNAMIC_OCT_TREE_MIN_VOLUME

template<class T> class InternalOctTree
{
protected:
	typedef std::unique_ptr<InternalOctTree<T>> MemberPointer;
	std::array<MemberPointer, 8> members;
	std::array<AABB, 8> internals;

	std::lsit<T> objects;
	const AABB bounds;
	const int depth;
public:
	InternalOctTree(const glm::vec3& negativeBound = glm::vec3(-DYNAMIC_OCT_TREE_DIMENSION),
		const glm::vec3& positiveBound = glm::vec3(DYNAMIC_OCT_TREE_DIMENSION), int depth = 0) noexcept
		: bounds(negativeBound, positiveBound), depth(depth)
	{
		this->Generate();
	};

	InternalOctTree(const AABB& bound, int depth = 0) noexcept : bounds(bound), depth(depth)
	{
		this->Generate();
	}

	~InternalOctTree() noexcept
	{
		// No need to clear as the list will clean naturally, and smart pointers will also be collected
		std::cout << "Cleaning Depth: " << this->depth << '\n';
	}

	void Clear() noexcept
	{
		this->objects.clear();
		for (MemberPointer& pointer : this->members)
		{
			pointer.reset();
		}
	}

	void Generate() noexcept
	{
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

	void Insert(const Object& obj, const AABB& box) noexcept
	{
		for (std::size_t i = 0; i < 8; i++)
		{
			glm::vec3 center = this->internals[i].GetCenter();
			glm::vec3 deviation = this->internals[i].Deviation();
			if (this->internals[i].Contains(box))
			{
				if (this->depth + 1 < DYANMIC_OCT_TREE_MAX_DEPTH)
				{
					if (!this->tree[i])
					{
						this->tree[i] = std::make_unique<InternalOctTree>(this->internals[i], depth + 1);
					}
					this->tree[i]->Insert(element, box);
					return;
				}
			}
		}
		this->objects.push_back({ element });
	}

	template<typename C>
	void Dump(C& container) const noexcept
	{
		std::copy(this->objects.begin(), this->objects.end(), std::back_inserter(container));
		for (const MemberPointer& pointer : this->members)
		{
			if (pointer)
			{
				pointer->Dump(container);
			}
		}
	}

	std::vector<T> Dump() const noexcept
	{
		std::vector<T> elements{ this->objects.size() };
		this->Dump(elements);
		return elements;
	}

	inline std::vector<T> InternalOctTree::Search(const AABB& box) const noexcept
	{
		std::vector<T> temp;
		return temp;
	}

	template<class C>
	inline void InternalOctTree::Search(const AABB& box, C& items) const noexcept
	{
		for (const auto& element : this->objects)
		{
			if (box.Overlap(element->GetAABB()))
				items.push_back(element);
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
};


template<class T> class DynamicOctTree
{
public:
	struct Elemental
	{
		T first;
		std::list<AABB, T*>::iterator second;
	};

	typedef std::list<T>::iterator Element;
	typedef std::pair<AABB, Element> ElementPair;
	typedef std::list<ElementPair>::iterator ListIterator;
	typedef std::pair<T, ListIterator> Object;
	typedef std::list<Object> Structure;

	template<typename F> 
		requires requires(F func, T& element)
	{
		{ func(element) } -> std::convertible_to<bool>;
	}
	void for_each(F operation)
	{
		for (Object& element : this->elements)
		{
			if (operation(element.first))
			{
				// reseat element
				AABB temp = element.second->first;
				std::remove(element.second, element.second + 1, [](const T& e) {return true; });
				this->root.Insert(element, temp);
			}
		}
	}

	Structure::iterator Insert(const T& element, const AABB& box) noexcept
	{
		Object& local = this->elements.emplace_back(element);

	}


	Structure::iterator begin() noexcept
	{
		return this->elements.begin();
	}

	Structure::const_iterator cbegin() const noexcept
	{
		return this->elements.cbegin();
	}

	Structure::iterator end() noexcept
	{
		return this->elements.end();
	}

	Structure::const_iterator cend() const noexcept
	{
		return this->elements.cend();
	}
	

protected:
	InternalOctTree<Structure::iterator> root;
	Structure elements;
};


#endif // DYNAMIC_OCT_TREE_H