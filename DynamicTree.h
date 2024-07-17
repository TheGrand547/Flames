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

// Will hold the index in the DynamicOctTree list of each element which is uhhh painfully linear
template<class T> class InternalOctTree
{
public:
	typedef std::pair<AABB, T> Element;
	typedef std::list<Element> Container;
protected:
	typedef std::unique_ptr<InternalOctTree<T>> MemberPointer;
	std::array<MemberPointer, 8> members;
	std::array<AABB, 8> internals;

	Container objects;
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

	typename Container::iterator Insert(const T& obj, const AABB& box) noexcept
	{
		for (std::size_t i = 0; i < 8; i++)
		{
			glm::vec3 center = this->internals[i].GetCenter();
			glm::vec3 deviation = this->internals[i].Deviation();
			if (this->internals[i].Contains(box))
			{
				if (this->depth + 1 < DYNAMIC_OCT_TREE_MAX_DEPTH)
				{
					if (!this->tree[i])
					{
						this->tree[i] = std::make_unique<InternalOctTree>(this->internals[i], depth + 1);
					}
					return this->tree[i]->Insert(obj, box);
				}
			}
		}
		this->objects.emplace_back(box, obj);
		return std::prev(this->objects.end());
	}

	template<typename C>
	void Dump(C& container) const noexcept
	{
		for (const auto& element : this->objects)
		{
			container.push_back(element.second);
		}
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
		std::vector<T> elements{};
		this->Dump(elements);
		return elements;
	}

	inline std::vector<T> Search(const AABB& box) const noexcept
	{
		std::vector<T> temp;
		this->Search(box, temp);
		return temp;
	}

	template<class C>
	void Search(const AABB& box, C& items) const noexcept
	{
		for (const auto& element : this->objects)
		{
			if (box.Overlap(element.first))
				items.push_back(element.second);
		}
		for (std::size_t i = 0; i < 8; i++)
		{
			if (this->members[i])
			{
				if (box.Contains(this->internals[i]))
				{
					this->members[i]->Dump(items);
				}
				else if (this->internals[i].Overlap(box))
				{
					this->members[i]->Search(box, items);
				}
			}
		}
	}
};

template<class T> class DynamicOctTree
{
public:
	typedef unsigned int Index;
	typedef std::list<std::pair<AABB, Index>> MemberType;
	typedef std::pair<T, MemberType::iterator> MemberPair;
	typedef std::vector<MemberPair> Structure;
	typedef Structure::iterator iterator;
	typedef Structure::const_iterator const_iterator;

	/*
	struct iterator
	{
	protected:
		Structure::iterator iter;
	public:
		constexpr iterator(const Structure::iterator& iter) noexcept : iter(iter) {}
		constexpr iterator(const iterator& iter) noexcept : iter(iter.iter) {}
		constexpr ~iterator() {}
		constexpr iterator& operator+=(const std::size_t i) noexcept
		{
			this->iter += i;
			return *this;
		}
		constexpr iterator operator+(const std::size_t i) const noexcept
		{
			return iterator(this->iter + i);
		}
		constexpr iterator& operator-=(const std::size_t i) noexcept
		{

		}
	};
	*/
	
	template<typename F> 
		requires requires(F func, T& element)
	{
		{ func(element) } -> std::convertible_to<bool>;
	}
	void for_each(F operation)
	{
		for (MemberPair& element : this->elements)
		{
			if (operation(element.element))
			{
				// reseat element
				AABB temp = element.second->first;
				Index index = element.second->second;
				std::remove(element.second, element.second + 1, [](const T& e) {return true; });
				this->root.Insert(index, temp);
			}
		}
	}

	std::vector<iterator> Search(const AABB& area) noexcept
	{
		std::vector<Index> index = this->root.Search(area);
		std::vector<iterator> value{index.size()};
		iterator start = this->elements.begin();
		for (Index i : index)
		{
			value.push_back(start + i);
		}
		return value;
	}

	std::vector<const_iterator> Search(const AABB& area) const noexcept
	{
		std::vector<Index> index = this->root.Search(area);
		std::vector<const_iterator> value{ index.size() };
		const_iterator start = this->elements.cbegin();
		for (Index& i : index)
		{
			value.push_back(start + i);
		}
		return value;
	}

	// Super expensive, try not to do this if you can avoid it
	void ReSeat()
	{
		QUICKTIMER(std::format("Reseating Tree with {} nodes", this->elements.size()));
		this->root.Clear();
		Index index = 0;
		for (const Structure::value_type& element : this->elements)
		{
			this->InternalInsert(index++, GetAABB(element.first));
		}
	}

	void ReserveSizeExact(const std::size_t& size) noexcept
	{
		this->root.Clear();
		this->elements.clear();
		this->elements.reserve(size);
	}

	// Reserves 50% more than you explicitly requires 
	void ReserveSize(const std::size_t size) noexcept
	{
		this->ReserveSizeExact((size * 3) >> 2);
	}

	iterator Insert(const T& element, const AABB& box) noexcept
	{
		std::size_t oldSize = this->elements.capacity();
		//Object& local = this->elements.emplace_back(element);
		if (this->elements.capacity() != oldSize)
		{
			this->ReSeat();
		}
		return this->elements.end();
	}

	iterator begin() noexcept
	{
		return this->elements.begin();
	}

	const_iterator cbegin() const noexcept
	{
		return this->elements.cbegin();
	}

	iterator end() noexcept
	{
		return this->elements.end();
	}

	const_iterator cend() const noexcept
	{
		return this->elements.cend();
	}
	

protected:
	// Index
	InternalOctTree<Index> root;
	Structure elements;

	void InternalInsert(Index index, const AABB& box)
	{
		// TODO: do it
	}
};


#endif // DYNAMIC_OCT_TREE_H