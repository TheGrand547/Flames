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
	{ std::equal_to<T>{}(node, node2) } -> std::convertible_to<bool>;
	{ node.distance(node2) } -> std::convertible_to<float>;
	{ node.neighbors() } -> std::convertible_to<std::vector<std::weak_ptr<T>>>;

	node == node2;
	std::equal_to<T>{}(node, node2);
	std::hash<T>{}(node);
	node.distance(node2);
};

template<typename T, typename W> concept HeuristicFunction = requires(T func, const W& a, const W& b)// std::convertible_to<std::function<float(const W&, const W&)>, T>;
{
	{ func(a, b) } -> std::convertible_to<float>;
};

template<typename T> using Heuristic_ = float (*)(const T&, const T&);

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

template<typename T, typename S = float> struct MinHeapValue
{
	T element;
	S value;
	constexpr MinHeapValue(T element, S value) : element(element), value(value) {}
	constexpr ~MinHeapValue() {}

	constexpr bool operator<(const MinHeapValue<T, S>& other) const noexcept
	{
		return other.value < this->value;
	}
};


/*
template<SearchNode Node> using SmartSearchNode = std::shared_ptr<Node>;
template<SearchNode Node> using WeakSearchNode = std::weak_ptr<Node>;
*/
// TODO: [[nodiscard]]
template<SearchNode Node, typename T>
	requires HeuristicFunction<T, Node>
std::pair<std::vector<std::shared_ptr<Node>>, std::unordered_set<std::shared_ptr<Node>>> AStarSearch(const std::shared_ptr<Node>& start,
						const std::shared_ptr<Node>& target, T heuristic)
{
	struct Scoring { float score = std::numeric_limits<float>::infinity(); };
	using SmartSearchNode = std::shared_ptr<Node>;
	using NodeMap = std::unordered_map<SmartSearchNode, SmartSearchNode>;
	using ScoreMap = std::unordered_map<SmartSearchNode, Scoring>;

	std::vector<MinHeapValue<SmartSearchNode>> openSet{ {start, 0.f} };
	std::unordered_set<SmartSearchNode> closedSet;

	NodeMap pathHistory{};

	// gScore
	ScoreMap cheapestPath{};
	cheapestPath[start].score = 0;

	std::vector<SmartSearchNode> finalPath;
	// fScore
	ScoreMap bestGuess;
	bestGuess[start].score = heuristic(*start, *target);
	while (openSet.size() > 0)
	{
		std::pop_heap(openSet.begin(), openSet.end());
		SmartSearchNode current = openSet.back().element;
		openSet.pop_back();
		if (closedSet.find(current) != closedSet.end())
		{
			std::cout << "Repeat detected" << std::endl;
			continue;
		}
		if (current == target)
		{
			while (current)
			{
				finalPath.push_back(current);
				current = pathHistory[current];
			}
			// Has to be a better way!
			std::reverse(finalPath.begin(), finalPath.end());
			break;
		}
		// Front is removed from open
		for (auto& neighbor : current->neighbors())
		{
			SmartSearchNode local = neighbor.lock();
			if (!local)
				continue;
			float tentative = cheapestPath[current].score + current->distance(*local);
			if (tentative < cheapestPath[local].score)
			{
				float oldGuess = bestGuess[local].score;
				pathHistory[local] = current;
				
				float newGuess = tentative + heuristic(*local, *target);
				cheapestPath[local].score = tentative;
				bestGuess[local].score = newGuess;

				openSet.push_back({ local, newGuess });
				std::push_heap(openSet.begin(), openSet.end());
				// Duplicates *are* inserted, but we hope that they will have a high enough key value to not be sorted towards
				// The front, simple hash key sort is performed to ensure they aren't re-explored
			}
		}
		closedSet.insert(current);
	}
	return std::make_pair(finalPath, closedSet);
}

#endif // PATHFINDING_H
