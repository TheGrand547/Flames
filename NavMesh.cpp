#include "NavMesh.h"
#include <filesystem>
#include <string>
#include <fstream>
#include "StaticVector.h"
#include "glmHelp.h"
#include "log.h"
#include "QuickTimer.h"

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
	std::cout << "Small Nodes: " << this->nodes.size() << '\n';
	{
		QUICKTIMER("Node Connections");
		StaticVector<std::vector<IndexType>> woof(this->nodes.size());
		for (auto i = 0; i < this->nodes.size(); i++)
		{
			Node& right = this->nodes[i];
			for (auto j = i; j < this->nodes.size(); j++)
			{
				Node& left = this->nodes[j];
				if (function(left, right))
				{
					woof[i].push_back(j);
					woof[j].push_back(i);
				}
			}
			right.connections.make(woof[i]);
		}
	}
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
			if (connectionNumber > 0)
			{
				local.connections.reserve(connectionNumber);
				input.read(reinterpret_cast<char*>(local.connections.begin()), sizeof(IndexType) * connectionNumber);
			}
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