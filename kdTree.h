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
	
	std::vector<glm::vec3> positions;
	std::vector<T> elements;
	
	void PropogateSubtree(std::span<T> elements, std::size_t start, Axes axis = Axes::X) noexcept
	{
		// TODO: customizeable median finding stuff;


		if (elements.size() > 1)
		{
			// TODO: Figure out if doing nth element thing will be sufficient

			/*
			std::vector<T> medianSet;
			std::size_t medianSetSize = std::min(elements.size(), 8); // TODO: Do tests with this to determine balance
			
			for (std::size_t i = 0; i < medianSetSize; i++)
			{

			}
			*/
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


			this->positions[start] = elements[size / 2]->GetPos();
			this->elements[start] = std::move(elements[size / 2]);

			this->PropogateSubtree(elements.subspan(0, size / 2), leftChild(start), next(axis));
			this->PropogateSubtree(elements.subspan(size / 2 + 1, size - (size / 2 + 1)), rightChild(start), next(axis));
		}
		else if (elements.size() == 1)
		{
			this->positions[start] = elements.front()->GetPos();
			this->elements[start] = elements.front();
		}
	}

public:
	kdTree() noexcept {}
	~kdTree() noexcept = default;

	// Invalidates internal state
	void Generate(std::vector<T>& elements)
	{
		// Unfortunate but has to be done
		std::size_t size = std::exp2(std::ceil(std::log2(elements.size())));

		// Stupid but it seems to work
		std::vector<T> works{ size };
		std::vector<glm::vec3> works2{ size, glm::vec3(NAN)};
		this->elements.swap(works);
		this->positions.swap(works2);
		this->PropogateSubtree(std::span{ elements }, 0);
		elements.clear();
	}

	void Print(std::size_t index = 0, std::size_t indent = 0) const
	{
		if (index < this->elements.size())
		{
			if (!glm::any(glm::isnan(this->positions[index])))
			{
				std::cout << std::setw(indent * 4) << ' ';
				std::cout << this->positions[index] << '\n';
				this->Print(leftChild(index), indent + 1);
				this->Print(rightChild(index), indent + 1);
			}
		}
	}
};


#endif // K_D_TREE_H