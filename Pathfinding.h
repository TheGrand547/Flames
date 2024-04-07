#pragma once
#ifndef PATHFINDING_H
#define PATHFINDING_H
#include <iostream>
#include <map>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>


template<typename T> concept SearchNode = requires(const T& node, const T& node2)
{
	{ std::hash<T>{}(node) } -> std::convertible_to<std::size_t>;
	{ std::distance(node, node2) } -> std::convertible_to<float>;
	{ std::equal_to(node, node2) } -> std::convertible_to<bool>;
	node == node2;
	std::equal_to(node, node2);
	std::hash<T>{}(node);
	std::distance(node, node2);

	{ node.neighbors() } -> std::convertible_to<std::span<std::weak_ptr<T>>>;
	// Neighbors
};

template<typename T> using Heuristic = float (*)(const T&, const T&);

template<typename T, typename S = float> struct MaxHeapValue
{
	T element;
	S value;
	constexpr MaxHeapValue(T element, S value) : element(element), value(value) {}
	constexpr ~MaxHeapValue() {}

	constexpr bool operator<(const MaxHeapValue<T, S>& other) const noexcept
	{
		return this->value < other.value;
	}
};

template<typename T, typename S = float> struct MinHeapValue : public MaxHeapValue<T, S>
{
	constexpr MinHeapValue(T element, S value) : MaxHeapValue<T, S>(element, value) {}
	constexpr ~MinHeapValue() {}

	constexpr bool operator<(const MinHeapValue<T, S>& other) const noexcept
	{
		return static_cast<MaxHeapValue<T, S>>(other) < static_cast<MaxHeapValue<T, S>>(*this);
	}
};


template<SearchNode Node>
std::vector<Node> AStarSearch(const Node& start, const Node& target, Heuristic<Node> heuristic)
{
	struct Scoring { float score = std::numeric_limits<float>::infinity; };
	using NodeMap = std::unordered_map<Node*, Node*>;
	using ScoreMap = std::unordered_map<Node*, Scoring>;

	// TODO: Proper thingy
	//std::priority_queue<MaxHeapValue> open{ {start, 0.f} };
	std::vector<MinHeapValue> openSet;

	std::unordered_set<Node> closedSet;

	// std::make_heap
	// std::push_heap

	NodeMap pathHistory{};

	// gScire
	ScoreMap cheapestPath{ {&start, 0} };
	
	// fScore
	ScoreMap bestGuess{ {&start, heuristic(start)} };
	while (openSet.size() > 0)
	{
		Node& current = std::pop_heap(openSet.begin(), openSet.end());
		if (closedSet.find(current) != closedSet.end())
		{
			std::cout << "Repeat detected" << std::endl;
			continue;
		}
		if (current == target)
		{
			// TODO: Reconstruct
			std::vector<Node> nodes;

		}
		// Front is removed from open
		for (auto& neighbor : current.neighbors())
		{
			float tentative = cheapestPath[current].score + std::distance(current, neighbor);
			if (tentative < cheapestPath[neighbor].score)
			{
				float oldGuess = bestGuess[neighbor];
				pathHistory[neighbor] = current;
				
				float newGuess = tentative + heuristic(neighbor, target);
				cheapestPath[neighbor].score = tentative;
				bestGuess[neighbor].score = newGuess;

				openSet.push_back({ neighbor, newGuess });
				std::push_heap(openSet.begin(), openSet.end());
				// Duplicates *are* inserted, but we hope that they will have a high enough key value to not be sorted towards
				// The front, simple hash key sort is performed to ensure they aren't re-explored
			}
		}
		closedSet.insert(current);
	}
}

#endif // PATHFINDING_H
