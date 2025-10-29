#pragma once
#ifndef DYNAMIC_OCT_TREE_H
#define DYNAMIC_OCT_TREE_H
#include <list>
#include <vector>
#include <algorithm>
#include "AABB.h"
#include "Log.h"
#include "QuickTimer.h"
#include "Lines.h"

constexpr auto DYNAMIC_OCT_TREE_MAX_DEPTH = (7);
constexpr auto DYNAMIC_OCT_TREE_MIN_RECURSIVE = (8);
constexpr auto DYNAMIC_OCT_TREE_DIMENSION = (100.f);
constexpr auto DYNAMIC_OCT_TREE_MIN_VOLUME = (10.f);

enum DynamicTreeEnum
{
	RESEAT, REMOVE, DO_NOTHING
};

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
		std::size_t size;
	public:
		InternalOctTree(const glm::vec3& negativeBound = glm::vec3(-DYNAMIC_OCT_TREE_DIMENSION),
			const glm::vec3& positiveBound = glm::vec3(DYNAMIC_OCT_TREE_DIMENSION), int depth = 0) noexcept
			: bounds(negativeBound, positiveBound), depth(depth), size(0)
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

		// Provides a strict upper bound on the number of elements in this tree, and its descenents
		std::size_t Size() const noexcept
		{
			return this->size;
		}

		void Generate() noexcept
		{
			glm::vec3 center  = this->bounds.GetCenter();
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
			this->size++;
			if (this->depth + 1 < DYNAMIC_OCT_TREE_MAX_DEPTH 
				&& this->size > DYNAMIC_OCT_TREE_MIN_RECURSIVE)
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

		std::size_t Recalculate() noexcept
		{
			this->size = this->objects.size();
			for (MemberPointer& pointer : this->members)
			{
				if (pointer)
				{
					std::size_t result = pointer->Recalculate();
					if (result == 0)
					{
						//pointer.reset(nullptr);
					}
					this->size += result;
				}
			}
			return this->size;
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
				if (pointer && pointer->size > 0)
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
				{
					items.push_back(element.second);
				}
			}
			for (std::size_t i = 0; i < 8; i++)
			{
				if (this->members[i] && this->members[i]->size > 0)
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

		void GetBoxes(std::vector<AABB>& staticBoxes) const noexcept
		{
			if (this->objects.size() > 0)
			{
				staticBoxes.push_back(this->bounds);
			}
			for (const MemberPointer& point: this->members)
			{
				if (point && point->size > 0)
				{
					point->GetBoxes(staticBoxes);
				}
			}
		}

		void Erase(Container::iterator iter) noexcept
		{
			// size must never be zero while there is a possibility of a child node having children
			// Thus the size must be greater than one before we're allowed to decrement it
			if (this->objects.size() > 1)
			{
				this->size--;
			}
			this->objects.erase(iter);
		}

		std::vector<Index> RayCast(Ray ray) const noexcept
		{
			std::vector<Index> hits;
			this->RayCast(ray, hits);
			// In case the top level tree isn't hit by the ray, it holds all the remaining elements, so make sure they're targetted
			if (!this->bounds.FastIntersect(ray) && false)
			{
				for (const auto& element : this->objects)
				{
					if (element.first.FastIntersect(ray))
					{
						hits.push_back(element.second);
					}
				}
			}
			return hits;
		}

		void RayCast(Ray ray, std::vector<Index>& out) const noexcept
		{
			if (this->bounds.FastIntersect(ray))
			{
				for (const auto& element : this->objects)
				{
					if (element.first.FastIntersect(ray))
					{
						out.push_back(element.second);
					}
				}
				for (std::size_t i = 0; i < 8; i++)
				{
					if (this->members[i] && this->members[i]->size > 0)
					{
						this->members[i]->RayCast(ray, out);
					}
				}
			}
		}

		bool QuickTest(const AABB& bounds) const noexcept
		{
			if (this->bounds.Overlap(bounds))
			{
				for (std::size_t i = 0; i < 8; i++)
				{
					if (this->members[i] && 
						this->members[i]->size > 0 && 
						this->members[i]->QuickTest(bounds))
					{
						return true;
					}
				}
				for (const auto& element : this->objects)
				{
					if (element.first.Overlap(bounds))
					{
						return true;
					}
				}
			}
			return false;
		}
};
	
public: 

	DynamicOctTree() noexcept : root() {}
	DynamicOctTree(const AABB& bounds) noexcept : root(bounds) {}
	~DynamicOctTree() noexcept = default;

	using Index = unsigned int;
	using MemberType = std::list<std::pair<AABB, Index>>;
	using MemberPair = std::pair<T, Member>;
	using Structure  = std::vector<MemberPair>;
	//typedef Structure::iterator iterator;
	//typedef Structure::const_iterator const_iterator;

	// Don't know if this is worth the effort
	
	template<typename Value>
	struct iteratorz
	{
	protected:
		Structure::iterator iter;
		//friend void DynamicOctTree<T>::Erase(DynamicOctTree<T>::iterator element) noexcept;
		friend DynamicOctTree<T>;
	public:
		using difference_type = std::ptrdiff_t;
		using value_type = Value;
		constexpr iteratorz() noexcept : iter() {}
		constexpr iteratorz(const Structure::iterator& iter) noexcept : iter(iter) {}
		constexpr iteratorz(const iteratorz& iter) noexcept : iter(iter.iter) {}
		constexpr ~iteratorz() noexcept {}
		constexpr iteratorz& operator+=(const std::size_t i) noexcept
		{
			this->iter += i;
			return *this;
		}
		constexpr iteratorz& operator-=(const std::size_t i) noexcept
		{
			this->iter -= i;
			return *this;
		}

		constexpr iteratorz operator+(const std::size_t i) const noexcept
		{
			return iteratorz(this->iter + i);
		}

		constexpr iteratorz& operator++() noexcept
		{
			this->iter++;
			return *this;
		}

		constexpr iteratorz operator++(int) noexcept
		{
			iteratorz old = *this;
			this->operator++();
			return old;
		}

		constexpr iteratorz& operator--() noexcept
		{
			--this->iter;
			return *this;
		}

		constexpr iteratorz operator--(int) noexcept
		{
			iteratorz old = *this;
			this->operator--();
			return old;
		}

		constexpr Value& operator*() const noexcept
		{
			return this->iter->first;
		}

		constexpr Value* operator->() const noexcept
		{
			return &this->iter->first;
		}

		constexpr bool operator==(const iteratorz& other) const noexcept
		{
			return this->iter == other.iter;
		}

		constexpr bool operator!=(const iteratorz& other) const noexcept = default;

		void swap(iteratorz<T> other) noexcept
		{
			std::iter_swap(this->iter, other.iter);
			std::iter_swap(this->iter->second.iterator, other.iter->second.iterator);
		}
	};
	static_assert(std::bidirectional_iterator<iteratorz<T>>);
	static_assert(std::bidirectional_iterator<iteratorz<const T>>);
	
	using value_type = T ;
	using iterator = iteratorz<T>;
	using const_iterator = iteratorz<const T>;
	using reverse_iterator = std::reverse_iterator<iteratorz<T>>;
	using const_reverse_iterator = std::reverse_iterator<iteratorz<const T>>;
	using reference = T&;

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
#endif // DEBUG
		const AABB current = GetAABB(element->first);

		// See if the current AABB is wholly contained in the same box as before
		if (element->second.pointer->bounds.Contains(current))
		{
			InternalOctTree* currentNode = element->second.pointer;
			// If it is, only update the bounding box in the structure, don't re-insert
			// Although this *could* lead to clogging up the top level of the structure but would need
			// to test that to verify. Could also insert only into that element so hmm, much to think on
			element->second.iterator->first = current;
			
			if (currentNode->depth < DYNAMIC_OCT_TREE_MAX_DEPTH && currentNode->Size() > DYNAMIC_OCT_TREE_MIN_RECURSIVE)
			{
				// Check if this fucks everything up
				Index index = element->second.iterator->second;
				currentNode->objects.erase(element->second.iterator);
				element->second = currentNode->Insert(index, current);
			}

#ifdef DEBUG
			total++;
			small += currentNode == element->second.pointer;
#endif // DEBUG
		}
		else
		{
			Index index = element->second.iterator->second;
			element->second.pointer->objects.erase(element->second.iterator);
			element->second = this->root.Insert(index, current);
		}
#ifdef DEBUG
		if (total % 10000 == 0)
		{
			Log(std::format("Ratio of partial to full insertions: {} : {}", small, total));
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

	// Returns the number of removed elements
	template<typename F> std::size_t EraseIf(F func)
	{
		std::size_t size = 0;
		//using iter_type = Structure::iterator;
		using iter_type = iterator;
		if (this->elements.size() == 0)
		{
			return size;
		}
		if (this->elements.size() == 1)
		{
			if (func(this->elements.front().first))
			{
				// CLEAN UP
				this->InternalErase(this->elements.begin());
				return 1;
			}
			return 0;
		}
		iter_type endOfInvalid = iterator(std::prev(this->elements.end()));
		bool clearAll = false;
		for (iter_type i = iterator(endOfInvalid); ; i--)
		{
			if (func(*i))
			{
				// Need to replace swap it to the 'start' of the 'invalid' region;
				size++;
				i.swap(endOfInvalid);
				endOfInvalid.iter->second.pointer->Erase(endOfInvalid.iter->second.iterator);
				if (endOfInvalid != this->elements.begin())
				{
					endOfInvalid--;
				}
				else
				{
					clearAll = true;
				}
			}
			if (i == this->elements.begin())
			{
				break;
			}
		}
		if (clearAll)
		{
			this->elements.clear();
		}
		else
		{
			this->elements.erase(endOfInvalid.iter + 1, this->elements.end());
		}
		// Make sure this gets rid of the pointer things
		return size;
	}

	// Returns the number of removed elements
	template<typename F>
		requires requires(F func, T& element)
	{
		{ func(element) } -> std::convertible_to<DynamicTreeEnum>;
	}
	std::size_t FullService(F func)
	{
		std::size_t size = 0;
		using iter_type = iterator;
		if (this->elements.size() == 0)
		{
			return 0;
		}
		if (this->elements.size() == 1)
		{
			auto mini = func(this->elements.front().first);
			switch (mini)
			{
			case REMOVE:
				// CLEAN UP
				this->InternalErase(this->elements.begin());
				return 1;
			case RESEAT:
				this->ReSeat(this->elements.begin());
				[[fallthrough]];
			default:
				return 0;
				
			}
		}
		iter_type endOfInvalid = iterator(std::prev(this->elements.end()));
		bool clearAll = false;
		for (iter_type i = iterator(endOfInvalid); ; i--)
		{
			DynamicTreeEnum local = func(*i);
			switch (local)
			{
				case REMOVE:
					{
						// Need to replace swap it to the 'start' of the 'invalid' region;
						size++;
						i.swap(endOfInvalid);
						endOfInvalid.iter->second.pointer->Erase(endOfInvalid.iter->second.iterator);
						if (endOfInvalid != this->elements.begin())
						{
							endOfInvalid--;
						}
						else
						{
							clearAll = true;
						}
						break;
					}
				case RESEAT:
					this->ReSeat(i.iter);
					break;
				default:
					break;
			}
			if (i == this->elements.begin())
			{
				break;
			}
		}
		if (clearAll)
		{
			this->elements.clear();
		}
		else
		{
			this->elements.erase(endOfInvalid.iter + 1, this->elements.end());
		}
		if (size > 0)
		{
			this->UpdateStructure();
		}
		// Make sure this gets rid of the pointer things
		return size;
	}

	void UpdateStructure() noexcept
	{
		this->root.Recalculate();
	}

	// Does *not* necessarily call the destructor for the element, or remove it from memory, it simply marks it as inactive
	void Erase(iterator element) noexcept
	{
		this->InternalErase(element.iter);
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

	std::vector<iterator> RayCast(Ray area) noexcept
	{
		std::vector<Index> index = this->root.RayCast(area);
		std::vector<iterator> value{};
		value.reserve(index.size());
		iterator start = this->elements.begin();
		for (Index i : index)
		{
			// Filthly stop-gap due to stupidity
			if (i >= this->elements.size())
			{
				Log("Undead member in Raycast results.");
				continue;
			}
			value.push_back(start + i);
		}
		return value;
	}

	// TODO: Figure out how to make this work
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
	template<typename Type> inline auto Search(const Type& type) const noexcept
	{
		return this->Search(type.GetAABB());
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
		this->ReserveSizeExact((size * 3) / 2);
	}

	iterator Insert(const T& element, const AABB& box) noexcept
	{
		std::size_t oldSize = this->elements.capacity();
		this->elements.emplace_back(element, Member{});
		this->InternalInsert(static_cast<Index>(this->elements.size() - 1), box);
		// I don't think this is necessary? 
		/*
		if (this->elements.capacity() != oldSize)
		{
			this->ReSeat();
		}
		*/
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
		std::vector<AABB> staticBoxes;
		this->root.GetBoxes(staticBoxes);
		return staticBoxes;
	}

	inline std::size_t size() const noexcept
	{
		return this->elements.size();
	}

	// Efficicently checks to see if *anything* in this overlaps the given bounds, trying to minimize comparisons
	bool QuickTest(const AABB& bounds) const noexcept
	{
		return this->root.QuickTest(bounds);
	}

	template<typename Type> bool QuickTest(const Type& type) const noexcept
	{
		return this->root.QuickTest(type.GetAABB());
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
			//temp->second.pointer->objects.erase(temp->second.iterator);
			iter->second.pointer->Erase(iter->second.iterator);
		}
		else
		{
			//iter->second.pointer->objects.erase(iter->second.iterator);
			iter->second.pointer->Erase(iter->second.iterator);
		}
		this->elements.pop_back();
	}
};


#endif // DYNAMIC_OCT_TREE_H