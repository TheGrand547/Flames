#pragma once
#ifndef K_D_TREE_H
#define K_D_TREE_H
#include <algorithm>
#include <span>
#include <vector>
#include "glmHelp.h"

// Axes are swapped because the Y axis filter is going to have a lot less splitting, leading to deeper trees
enum class Axes : unsigned char
{
	X = 0, Z = 1, Y = 2
};

constexpr Axes next(const Axes axis) noexcept
{
	return static_cast<Axes>((static_cast<unsigned char>(axis) + 1) % 3);
}

template<typename F> concept HasPosition = requires(const F& f)
{
	f.position;
	{ f.position } -> std::convertible_to<glm::vec3>;
};

// TODO: Make median function customizable

// TODO: *require* this to be a pointer and have a GetPos() function
template<class T> 
class kdTree
{
protected:
	T element;
	glm::vec3 pivot;
	Axes axis;
	bool valid = true;

	std::unique_ptr<kdTree> left = nullptr, right = nullptr;
	std::size_t count;

	template<typename F> static inline glm::vec3 GetPos(const F& f) noexcept
		requires (HasPosition<F>)
	{
		return f.position;
	}

	template<typename F> static inline glm::vec3 GetPos(const F& f) noexcept
	{
		return f->GetPos();
	}

	// ????
	//void PropogateSubtree(std::span<T> elements, std::size_t start, Axes axis = Axes::X) noexcept

	kdTree(T element, Axes axis) noexcept : element(element), pivot(GetPos<T>(element)), axis(axis), count(1) {}

	// TODO: ITERATOR
	/*
	template<typename P>
	class _iterator
	{
	private:
		kdTree<P>* parent = nullptr;
		bool visitedLeft = false, visitedRight = false;
	public:

		_iterator& operator++() noexcept
		{

		}
	};
	*/
	void nearestNeightborInternal(const glm::vec3& point, float& smallestDistance, T const *& currentBest, float minimumRadius) const noexcept
	{
		float myDistance = glm::distance2(point, this->pivot);
		// This is a leaf node
		if (!this->left && !this->right)
		{
			if (minimumRadius < myDistance && myDistance < smallestDistance)
			{
				currentBest = &this->element;
				smallestDistance = myDistance;
			}
			return;
		}
		float myComponent    = this->pivot[static_cast<unsigned char>(this->axis)];
		float pointComponent = point[static_cast<unsigned char>(this->axis)];

		bool leftComparison  = pointComponent <= myComponent;
		bool rightComparison =    myComponent <= pointComponent;

		bool notBoth = !(leftComparison && rightComparison);

		// Ensure children get ignored if the parent is closer
		if (minimumRadius < myDistance && myDistance < smallestDistance)
		{
			currentBest = &this->element;
			smallestDistance = myDistance;
		}

		// Check the part of the tree that the point would be in, if it was being inserted
		if (leftComparison && this->left)
		{
			this->left->nearestNeightborInternal(point, smallestDistance, currentBest, minimumRadius);
		}
		if (rightComparison && this->right)
		{
			this->right->nearestNeightborInternal(point, smallestDistance, currentBest, minimumRadius);
		}
		// If both branches were checked already don't waste time re-checking them
		if (notBoth)
		{
			// Operating on squared distances
			float distance = glm::abs(pointComponent - myComponent);
			distance *= distance;
			// If it's smaller then that's bad
			if (minimumRadius < distance && distance < smallestDistance)
			{
				// If we checked the left branch before then we must check the right branch
				if (leftComparison && this->right)
				{
					this->right->nearestNeightborInternal(point, smallestDistance, currentBest, minimumRadius);
				}
				// Similarly here
				if (rightComparison && this->left)
				{
					this->left->nearestNeightborInternal(point, smallestDistance, currentBest, minimumRadius);
				}
			}
		}
	}

	void neighborsInRangeInternal(const glm::vec3& center, std::vector<T>& elements, float outerRadius, float innerRadius) const noexcept
	{
		float myDistance = glm::distance2(center, this->pivot);
		if (innerRadius < myDistance && myDistance < outerRadius)
		{
			elements.push_back(this->element);
		}

		float myComponent    = this->pivot[static_cast<unsigned char>(this->axis)];
		float pointComponent = center[static_cast<unsigned char>(this->axis)];

		bool leftComparison  = pointComponent <= myComponent;
		bool rightComparison =    myComponent <= pointComponent;

		float axisDistance = glm::abs(myComponent - pointComponent);
		axisDistance *= axisDistance;

		if (leftComparison && this->left)
		{
			this->left->neighborsInRangeInternal(center, elements, outerRadius, innerRadius);
		}
		if (rightComparison && this->right)
		{
			this->right->neighborsInRangeInternal(center, elements, outerRadius, innerRadius);
		}

		// Too close or too far
		if (outerRadius < axisDistance || axisDistance < innerRadius)
			return;
		if (leftComparison && rightComparison)
			return;
		if (rightComparison && this->left)
		{
			this->left->neighborsInRangeInternal(center, elements, outerRadius, innerRadius);
		}
		if (leftComparison && this->right)
		{
			this->right->neighborsInRangeInternal(center, elements, outerRadius, innerRadius);
		}
	}

	void leftInsert(T&& element) noexcept
	{
		if (this->left)
		{
			this->left->insert(std::forward(element));
			this->count++;
		}
		else
		{
			this->left = std::make_unique<kdTree<T>>(element, next(this->axis));
			this->count += this->left->count;
		}
	}

	void rightInsert(T&& element) noexcept
	{
		if (this->right)
		{
			this->right->insert(std::forward(element));
			this->count++;
		}
		else
		{
			this->right = std::make_unique<kdTree<T>>(element, next(this->axis));
			this->count += this->right->count;
		}
	}

public:
	kdTree() noexcept : element(), pivot(666), axis(Axes::X), count(0) {}
	kdTree(std::span<T> elements, Axes axis) noexcept : element(), pivot(0), axis(axis), count(1)
	{
		if (elements.size() > 1)
		{
			// TODO: Figure out if doing nth element thing will be sufficient
			const std::size_t size = elements.size();
			const std::size_t half = size / 2;
			std::nth_element(elements.begin(), elements.begin() + half, elements.end(),
				[axis](const T& left, const T& right)
				{
					return GetPos<T>(left)[static_cast<unsigned char>(axis)] < GetPos<T>(right)[static_cast<unsigned char>(axis)];
				}
			);
			// [000]1[000] 
			// size = 7
			// 7 / 2 => 3 is median
			// 7 / 2 => 3 is number of elements in first subspan
			// 7 / 2 + 1 => 4 is the location of the second subspan
			// ceil(7 / 2) - 1 => 3 is number of elements in the second subspan
			// 
			// [0000]1[000]
			// size = 8
			// 8 / 2 => 4 is median
			// 8 / 2 => 4 is number of elements in the first subspan
			// 8 / 2 + 1 => 5 is the index of the second subspan
			// ceil(8 / 2) - 1 => 3 is the number of elements in the second subspan
			// This is the median

			this->element = elements[half];
			this->pivot = GetPos<T>(elements[half]);

			std::span<T> left  = elements.subspan(0, half);
			std::span<T> right = elements.subspan(half + 1, size - (half + 1));
			this->count = size;
			if (!left.empty()) 
			{
				this->left = std::make_unique<kdTree<T>>(left, next(this->axis));
			}
			if (!right.empty()) 
			{
				this->right = std::make_unique<kdTree<T>>(right, next(this->axis));
			}

		}
		else if (elements.size() == 1)
		{
			this->pivot = GetPos<T>(elements.front());
			this->element = elements.front();
		}
	}

	void Clear() noexcept
	{
		if (this->right) this->right.reset();
		if (this->left) this->left.reset();
		this->valid = false;
	}

	bool Valid() const noexcept
	{
		return this->valid;
	}
	
	//typedef iterator _iterator<T>;
	//typedef const_iterator _iterator<T>;

	kdTree<T>& operator=(kdTree<T>&& other) noexcept
	{
		this->element = std::move(other.element);
		this->pivot = other.pivot;
		this->left = std::move(other.left);
		this->right = std::move(other.right);
		this->count = other.count;
		this->valid = true;
		other.valid = false;
		return *this;
	}

	// Invalidates internal state
	static kdTree<T> Generate(std::vector<T>& elements) noexcept
	{
		return kdTree(std::span{ elements }, Axes::X);
	}

	void Print(std::size_t indent = 0) const
	{
		//std::cout << std::setw(indent * 4) << ' ';
		if (this->left && this->right)
		{
			std::cout << std::setw(indent * 4) << ' ';
			std::cout << this->left->size() << ":" << this->right->size() << '\n';
			this->left->Print(indent + 1);
			this->right->Print(indent + 1);

		}
		//std::cout << this->pivot << '\n';
		//if (this->left) this->left->Print(indent + 1);
		//if (this->right) this->right->Print(indent + 1);
	}
	
	std::size_t size() const noexcept
	{
		return this->count;
	}

	// TODO: Determine if given the implied use case of shared_ptr the extra indirection is unhelpful
	const T& nearestNeighbor(const glm::vec3& point) const noexcept
	{
		T const* currentBest = &this->element;
		float currentDistance = INFINITY;
		this->nearestNeightborInternal(point, currentDistance, currentBest, -INFINITY);
		return *currentBest;
	}

	T const * nearestNeighbor(const glm::vec3& point, float minimumDistance) const noexcept
	{
		T const * currentBest = &this->element;
		float currentDistance = INFINITY;
		this->nearestNeightborInternal(point, currentDistance, currentBest, glm::abs(minimumDistance * minimumDistance));
		if (currentDistance < minimumDistance)
			currentBest = nullptr;
		return currentBest;
	}

	void neighborsInRange(std::vector<T>& output, const glm::vec3& center, float outerRadius, float innerRadius = -INFINITY) const noexcept
	{
		output.clear();
		this->neighborsInRangeInternal(center, output, glm::abs(outerRadius * outerRadius), glm::abs(innerRadius) * innerRadius);
	}

	std::vector<T> neighborsInRange(const glm::vec3& center, float outerRadius, float innerRadius = -INFINITY) const noexcept
	{
		std::vector<T> results;
		this->neighborsInRange(results, center, outerRadius, innerRadius);
		return results;
	}

	void insert(T&& element) noexcept
	{
		glm::vec3 point = GetPos<T>(element);
		float myComponent = this->pivot[static_cast<unsigned char>(this->axis)];
		float pointComponent = point[static_cast<unsigned char>(this->axis)];

		bool leftComparison = pointComponent <= myComponent;
		bool rightComparison = myComponent <= pointComponent;
		if (leftComparison && rightComparison)
		{
			if (this->left && this->right)
			{
				((this->left->size() < this->right->size()) ? this->left : this->right)->insert(std::forward(element));
			}
			else if (this->left)
			{
				this->rightInsert(std::forward(element));
			}
			else
			{
				this->leftInsert(std::forward(element));
			}
		}
		else if (rightComparison)
		{
			this->rightInsert(std::forward(element));
		}
		else // Left Comparison must be true or both comparisons will be invalid, and in that case just default to left
		{
			this->leftInsert(std::forward(element));
		}
	}
};


#endif // K_D_TREE_H