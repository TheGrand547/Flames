#pragma once
#ifndef K_D_TREE_H
#define K_D_TREE_H
#include <algorithm>
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

// TODO: Make median function customizable

// TODO: *require* this to be a pointer and have a GetPos() function
template<class T> 
class kdTree
{
protected:
	static constexpr std::size_t leftChild(const std::size_t i) noexcept
	{
		return (i * 2) + 1;
	}

	static constexpr std::size_t rightChild(const std::size_t i) noexcept
	{
		return (i * 2) + 2;
	}
	
	//std::vector<glm::vec3> positions;
	//std::vector<T> elements;
	
	T element;
	glm::vec3 pivot;
	Axes axis;

	std::unique_ptr<kdTree> left = nullptr, right = nullptr;

	//void PropogateSubtree(std::span<T> elements, std::size_t start, Axes axis = Axes::X) noexcept

	kdTree(T element, Axes axis) noexcept : element(element), pivot(element->GetPos()), axis(axis) {}

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
public:
	kdTree() noexcept : element(), pivot(666), axis(Axes::X) {}
	kdTree(std::span<T> elements, Axes axis) noexcept : element(), pivot(0), axis(axis)
	{
		if (elements.size() > 1)
		{
			// TODO: Figure out if doing nth element thing will be sufficient
			std::size_t size = elements.size();
			std::nth_element(elements.begin(), elements.begin() + (size / 2), elements.end(),
				[axis](const T& left, const T& right)
				{
					return left->GetPos()[static_cast<unsigned char>(axis)] < right->GetPos()[static_cast<unsigned char>(axis)];
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

			this->element = elements[size / 2];
			this->pivot = elements[size / 2]->GetPos();

			std::span<T> left = elements.subspan(0, size / 2);
			std::span<T> right = elements.subspan(size / 2 + 1, size - (size / 2 + 1));
			if (!left.empty()) this->left = std::make_unique<kdTree<T>>(left, next(this->axis));
			if (!right.empty()) this->right = std::make_unique<kdTree<T>>(right, next(this->axis));
		}
		else if (elements.size() == 1)
		{
			this->pivot = elements.front()->GetPos();
			this->element = elements.front();
		
		}
	}


	//typedef iterator _iterator<T>;
	//typedef const_iterator _iterator<T>;

	kdTree<T>& operator=(kdTree<T>&& other) noexcept
	{
		this->element = std::move(other.element);
		this->pivot = other.pivot;
		this->left = std::move(other.left);
		this->right = std::move(other.right);
		return *this;
	}

	// Invalidates internal state
	static kdTree<T> Generate(std::vector<T>& elements) noexcept
	{
		// Unfortunate but has to be done
		//std::size_t size = std::exp2(std::ceil(std::log2(elements.size())));

		// Stupid but it seems to work
		//std::vector<T> works{ size };
		//std::vector<glm::vec3> works2{ size, glm::vec3(NAN)};
		//this->elements.swap(works);
		//this->positions.swap(works2);
		//this->PropogateSubtree(std::span{ elements }, 0);

		//elements.clear();
		return kdTree(std::span{ elements }, Axes::X);
	}

	void Print(std::size_t indent = 0) const
	{
		std::cout << std::setw(indent * 4) << ' ';
		std::cout << this->pivot << '\n';
		if (this->left) this->left->Print(indent + 1);
		if (this->right) this->right->Print(indent + 1);
	}

	std::size_t size() const noexcept
	{
		std::size_t size = 1;
		if (this->left) size += this->left->size();
		if (this->right) size += this->right->size();
		return size;
	}


	void nearestNeightbor2(const glm::vec3& point, float& smallestDist, T*& currentBest) noexcept
	{
		if (!this->left && !this->right && glm::distance(point, this->pivot) < smallestDist)
		{
			currentBest = &this->element;
			smallestDist = glm::distance(point, this->pivot);
			return;
		}
		bool leftComparison  = glm::lessThanEqual(point, this->pivot)[static_cast<unsigned char>(this->axis)];
		bool rightComparison = glm::lessThanEqual(this->pivot, point)[static_cast<unsigned char>(this->axis)];
		if (leftComparison && this->left)
		{
			this->left->nearestNeightbor2(point, smallestDist, currentBest);
		}
		if (rightComparison && this->right)
		{
			this->right->nearestNeightbor2(point, smallestDist, currentBest);
		}
		if (glm::distance(point, this->pivot) < smallestDist)
		{
			currentBest = &this->element;
			smallestDist = glm::distance(point, this->pivot);
		}
		if (!(leftComparison && rightComparison))
		{
			float distance = glm::abs(point[static_cast<unsigned char>(this->axis)] - this->pivot[static_cast<unsigned char>(this->axis)]);
			// If it's smaller then that's bad
			if (distance < smallestDist)
			{
				if (leftComparison && this->right)
				{
					this->right->nearestNeightbor2(point, smallestDist, currentBest);
				}
				if (rightComparison && this->left)
				{
					this->left->nearestNeightbor2(point, smallestDist, currentBest);
				}
			}
		}
	}


	T& nearestNeighbor(const glm::vec3& point) noexcept
	{
		T* currentBest = &this->element;
		float currentDistance = INFINITY;
		this->nearestNeightbor2(point, currentDistance, currentBest);
		return *currentBest;
	}
};


#endif // K_D_TREE_H