#pragma once
#ifndef DYNAMIC_OCT_TREE_H
#define DYNAMIC_OCT_TREE_H
#include <list>
#include "AABB.h"
#include "Log.h"
#include "QuickTimer.h"

#ifndef DYNAMIC_OCT_TREE_MAX_DEPTH
#define DYNAMIC_OCT_TREE_MAX_DEPTH (8)
#endif //DYNAMIC_OCT_TREE_MAX_DEPTH

#ifndef DYNAMIC_OCT_TREE_DIMENSION
#define DYNAMIC_OCT_TREE_DIMENSION (100.f)
#endif // DYNAMIC_OCT_TREE_DIMENSION

#ifndef DYNAMIC_OCT_TREE_MIN_VOLUME
#define DYNAMIC_OCT_TREE_MIN_VOLUME (10.f)
#endif // DYNAMIC_OCT_TREE_MIN_VOLUME

template<class T> class DynamicOctTree
{
protected:
	struct InternalOctTree;
	struct Member
	{
		InternalOctTree* pointer = nullptr;
		typename std::list<std::pair<AABB, unsigned int>>::iterator iterator;
	};

	struct InternalOctTree
	{
		typedef unsigned int Index;
		typedef std::pair<AABB, Index> Element;
		typedef std::list<Element> Container;
		typedef std::unique_ptr<InternalOctTree> MemberPointer;

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

		Member Insert(const Index& obj, const AABB& box) noexcept
		{
			if (this->depth + 1 < DYNAMIC_OCT_TREE_MAX_DEPTH)
			{
				for (std::size_t i = 0; i < 8; i++)
				{
					if (this->internals[i].Contains(box))
					{
						if (!this->members[i])
						{
							this->members[i] = std::make_unique<InternalOctTree>(this->internals[i], depth + 1);
						}
						return this->members[i]->Insert(obj, box);
					}
				}
			}
			this->objects.emplace_back(box, obj);
			return { this, static_cast<typename Container::iterator>(std::prev(this->objects.end())) };
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

		std::vector<Index> Dump() const noexcept
		{
			std::vector<Index> elements{};
			this->Dump(elements);
			return elements;
		}

		inline std::vector<Index> Search(const AABB& box) const noexcept
		{
			std::vector<Index> temp;
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

		void GetBoxes(std::vector<AABB>& boxes) const noexcept
		{
			if (this->objects.size() > 0)
				boxes.push_back(this->bounds);
			for (const MemberPointer& point: this->members)
			{
				if (point)
				{
					point->GetBoxes(boxes);
				}
			}
		}
	};
	
public: 

	DynamicOctTree() noexcept : root() {}
	DynamicOctTree(const AABB& bounds) noexcept : root(bounds) {}
	~DynamicOctTree() noexcept = default;

	typedef unsigned int Index;
	typedef std::list<std::pair<AABB, Index>> MemberType;
	typedef std::pair<T, Member> MemberPair;
	typedef std::vector<MemberPair> Structure;
	typedef Structure::iterator iterator;
	typedef Structure::const_iterator const_iterator;

	// Don't know if this is worth the effort
	/*
	struct iteratorz
	{
	protected:
		Structure::iterator iter;
	public:
		constexpr iteratorz(const Structure::iterator& iter) noexcept : iter(iter) {}
		constexpr iteratorz(const iteratorz& iter) noexcept : iter(iter.iter) {}
		constexpr ~iteratorz() noexcept {}
		constexpr iteratorz& operator+=(const std::size_t i) noexcept
		{
			this->iter += i;
			return *this;
		}

		constexpr iteratorz operator+(const std::size_t i) const noexcept
		{
			return iteratorz(this->iter + i);
		}

		constexpr iteratorz& operator-=(const std::size_t i) noexcept
		{
			this->iter -= i;
			return *this;
		}
	};
	*/

	// Invalidates the whole tree
	void AdjustBounds(const AABB& box) noexcept
	{
		this->root.Clear();
		this->root.bounds = box;
		this->root.Generate();
	}

	void ReSeat(typename Structure::iterator element) noexcept
	{
#ifdef DEBUG
		static std::size_t total = 0;
		static std::size_t small = 0;
		total++;
#endif // DEBUG
		AABB current = GetAABB(element->first);

		// See if the current AABB is wholly contained in the same box as before
		if (element->second.pointer->bounds.Contains(current))
		{
			// If it is, only update the bounding box in the structure, don't re-insert
			// Although this *could* lead to clogging up the top level of the structure but would need
			// to test that to verify. Could also insert only into that element so hmm, much to think on
			element->second.iterator->first = current;
#ifdef DEBUG
			InternalOctTree* temp = element->second.pointer;
#endif // DEBUG

			// Check if this fucks everything up
			Index index = element->second.iterator->second;
			element->second.pointer->objects.erase(element->second.iterator);
			element->second = element->second.pointer->Insert(index, GetAABB(element->first));

#ifdef DEBUG
			small += temp == element->second.pointer;
#endif // DEBUG
		}
		else
		{
			Index index = element->second.iterator->second;
			element->second.pointer->objects.erase(element->second.iterator);
			element->second = this->root.Insert(index, GetAABB(element->first));
		}
#ifdef DEBUG
		if (total % 10000 == 0)
		{
			std::cout << small << ":" << total << std::endl;
		}
#endif // DEBUG
	}
	
	template<typename F> 
		requires requires(F func, T& element)
	{
		{ func(element) } -> std::convertible_to<bool>;
	}
	void for_each(F operation)
	{
		for (typename Structure::iterator element = this->elements.begin(); element != this->elements.end(); element++)
		{
			if (operation(element->first))
			{
				this->ReSeat(element);
			}
		}
	}

	void Erase(Structure::iterator element) noexcept
	{
		this->InternalErase(element);
	}

	std::vector<iterator> Search(const AABB& area) noexcept
	{
		std::vector<Index> index = this->root.Search(area);
		std::vector<iterator> value{};
		value.reserve(index.size());
		iterator start = this->elements.begin();
		for (Index i : index)
		{
			// Filthly stop-gap due to stupidity
			if (i >= this->elements.size())
			{
				Log("Undead member in search results.");
				continue;
			}
			value.push_back(start + i);
		}
		return value;
	}

	std::vector<const_iterator> Search(const AABB& area) const noexcept
	{
		std::vector<Index> index = this->root.Search(area);
		std::vector<const_iterator> value{};
		value.reserve(index.size());
		const_iterator start = this->elements.cbegin();
		for (Index& i : index)
		{
			value.push_back(start + i);
		}
		return value;
	}

	// (Probably) Super expensive, try not to do this if you can avoid it
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

	// Reserves 50% more than you explicitly request for safety from re-allocation 
	void ReserveSize(const std::size_t size) noexcept
	{
		this->ReserveSizeExact((size * 3) >> 2);
	}

	iterator Insert(const T& element, const AABB& box) noexcept
	{
		std::size_t oldSize = this->elements.capacity();
		this->elements.push_back({ element, Member{} });
		this->InternalInsert(static_cast<Index>(this->elements.size() - 1), box);
		if (this->elements.capacity() != oldSize)
		{
			this->ReSeat();
		}
		return std::prev(this->elements.end());
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
	
	std::vector<AABB> GetBoxes() const noexcept
	{
		std::vector<AABB> boxes;
		this->root.GetBoxes(boxes);
		return boxes;
	}

protected:
	// Index
	InternalOctTree root;
	Structure elements;

	void InternalInsert(Index index, const AABB& box) noexcept
	{
		this->elements[index].second = this->root.Insert(index, box);
	}

	void InternalErase(Structure::iterator iter) noexcept
	{
		if (std::next(iter) != this->elements.end())
		{
			typename Structure::iterator temp = std::prev(this->elements.end());
			Index stored = iter->second.iterator->second;
			std::iter_swap(temp, iter);
			iter->second.iterator->second = stored;
			temp->second.pointer->objects.erase(temp->second.iterator);
		}
		else
		{
			iter->second.pointer->objects.erase(iter->second.iterator);
		}
		this->elements.pop_back();
	}
};


#endif // DYNAMIC_OCT_TREE_H