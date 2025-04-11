#include "NavMesh.h"
#include <filesystem>
#include <string>
#include <fstream>
#include "StaticVector.h"
#include "glmHelp.h"
#include "log.h"
#include "QuickTimer.h"
#include <ranges>
#include "Parallel.h"
#include "Pathfinding.h"
#include <set>

static const std::string basePath = "Levels/";
static const std::string extension = ".nmbin";
static const std::string Validation = "NAV";

/*
 File format
* Three byte sequence: "NAV"
* Four byte unsigned int: Node Count
* Node Count number of nodes
* Node Format
* - glm::vec3 position
* - std::uint16_t type
* - unsigned int number of #connections
* - #connections unsigned int indexes
*/

NavMesh::NavMesh(std::string name) noexcept : name(name)
{
	
}

void NavMesh::Generate(std::span<const glm::vec3> points, std::function<bool(const Node&, const Node&)> function) noexcept
{
	this->nodes.clear();
	std::transform(points.begin(), points.end(), std::back_inserter(this->nodes),
		[](const glm::vec3 point) -> Node
		{
			return { point, 0, {} };
		}
	);
	IndexType connectionCount = 0;
	std::cout << "Small Nodes: " << this->nodes.size() << '\n';
	{
		QUICKTIMER("Node Connections");
		StaticVector<std::vector<IndexType>> woof(this->nodes.size());
		std::mutex mut;
		for (auto i = 0; i < this->nodes.size(); i++)
		{
			Node& right = this->nodes[i];
			std::ranges::iota_view viewing(static_cast<std::size_t>(i + 1), static_cast<std::size_t>(this->nodes.size()));

			// Toss up between par and par_unseq, will have to try with bigger datasets
			Parallel::for_each(std::execution::par_unseq, viewing.begin(), viewing.end(), [&](std::size_t j)
				{
					Node& left = this->nodes[j];
					if (function(left, right))
					{
						woof[j].push_back(i);
						{
							connectionCount++;
							std::lock_guard guard(mut);
							woof[i].push_back(static_cast<IndexType>(j));
						}
					}
				}
			);
			right.connections.make(woof[i]);
			std::sort(right.connections.begin(), right.connections.end());
		}
	}
	Log(std::format("Connections Made: {}", connectionCount));
}

bool NavMesh::Load(std::string filename) noexcept
{
	this->name = filename;
	return this->Load();
}

bool NavMesh::Load() noexcept
{
	std::filesystem::path target(basePath + this->name + extension);
	if (std::filesystem::exists(target))
	{
		QUICKTIMER("Loading");
		std::ifstream input;
		input.open(target, std::ios::binary);
		char buffer[3]{};
		input.read(buffer, 3);
		if (!strcmp(buffer, Validation.c_str()))
		{
			Log(std::format("'{}' is an invalid NavMesh file", this->name));
			return false;
		}
		IndexType count = 0;
		input.read(reinterpret_cast<char*>(&count), sizeof(IndexType));
		this->nodes.clear();
		this->nodes.reserve(count);
		for (IndexType i = 0; i < count; i++)
		{
			nodes.emplace_back();
			Node& local = nodes.back();
			input.read(reinterpret_cast<char*>(&local.position), sizeof(glm::vec3));
			IndexType connectionNumber = 0;
			input.read(reinterpret_cast<char*>(&local.type), sizeof(std::uint16_t));
			input.read(reinterpret_cast<char*>(&connectionNumber), sizeof(IndexType));
			//std::cout << connectionNumber << '\n';
			if (connectionNumber > 0)
			{
				local.connections.reserve(connectionNumber);
				input.read(reinterpret_cast<char*>(local.connections.begin()), sizeof(IndexType) * connectionNumber);
			}
			//std::cout << local.connections.size() << "\n\n";
		}
		input.close();
		return true;
	}
	else
	{
		Log(std::format("'{}' does not exist as a NavMesh file", this->name));
		return false;
	}
}

void NavMesh::Export() noexcept
{
	std::filesystem::path target(basePath + this->name + extension);
	std::ofstream output;
	output.open(target, std::ios::binary);
	if (!output.is_open())
	{
		Log(std::format("Failed to export structure to file '{}'", this->name + extension));
		return;
	}
	QUICKTIMER("Exporting");
	output.write(Validation.c_str(), Validation.length());
	IndexType elementCount = static_cast<IndexType>(nodes.size());
	output.write(reinterpret_cast<char*>(&elementCount), sizeof(IndexType));
	for (IndexType i = 0; i < elementCount; i++)
	{
		const auto& current = nodes[i];
		IndexType connectionCount = static_cast<IndexType>(current.connections.size());
		output.write(reinterpret_cast<const char*>(&current.position), sizeof(glm::vec3));
		output.write(reinterpret_cast<const char*>(&current.type),     sizeof(std::uint16_t));
		output.write(reinterpret_cast<const char*>(&connectionCount),  sizeof(IndexType));
		output.write(reinterpret_cast<const char*>(&*current.connections.cbegin()), sizeof(IndexType) * connectionCount);
	}
	output.close();
}

[[nodiscard]] std::vector<glm::vec3> NavMesh::AStar(IndexType start, IndexType target, 
	std::function<float(const Node&, const Node&)> heuristic) const noexcept
{
	struct Scoring { float score = std::numeric_limits<float>::infinity(); };
	using SmartSearchNode = IndexType;
	using NodeMap = std::map<SmartSearchNode, SmartSearchNode>;
	using ScoreMap = std::map<SmartSearchNode, Scoring>;

	std::vector<MinHeapValue<SmartSearchNode>> openSet{ {start, 0.f} };
	//std::unordered_set<SmartSearchNode> closedSet;
	std::set<SmartSearchNode> closedSet;
	QUICKTIMER("A* on NavMesh");
	NodeMap pathHistory{};

	// gScore
	ScoreMap cheapestPath{};
	cheapestPath[start].score = 0;

	std::vector<glm::vec3> finalPath;
	// fScore
	ScoreMap bestGuess;
	bestGuess[start].score = heuristic(this->nodes[start], this->nodes[target]);
	while (openSet.size() > 0)
	{
		std::pop_heap(openSet.begin(), openSet.end());
		SmartSearchNode current = openSet.back().element;
		openSet.pop_back();
		if (closedSet.find(current) != closedSet.end())
		{
			continue;
		}
		if (current == target)
		{
			while (current != start)
			{
				finalPath.push_back(this->nodes[current].position);
				current = pathHistory[current];
			}
			// Has to be a better way!
			//std::reverse(finalPath.begin(), finalPath.end());
			break;
		}
		const Node& normal = this->nodes[current];
		// Front is removed from open
		for (auto& neighbor : normal.connections)
		{
			const SmartSearchNode& local = neighbor;
			const Node& ref = this->nodes[local];
			float tentative = cheapestPath[current].score + normal.distance(ref);
			if (tentative < cheapestPath[local].score)
			{
				float oldGuess = bestGuess[local].score;
				pathHistory[local] = current;

				float newGuess = tentative + heuristic(ref, this->nodes[target]);
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
	Log(std::format("Nodes Explored: {}\tPath Length: {}", closedSet.size(), finalPath.size()));
	return finalPath;
}
