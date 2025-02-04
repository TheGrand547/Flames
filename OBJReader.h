#pragma once
#ifndef OBJ_READER_H
#define OBJ_READER_H
#include <string>
#include <vector>
#include "Buffer.h"

struct MeshPair
{
	ArrayBuffer vertex;
	ElementArray index;
};

struct OBJReader
{
	static void ReadOBJ(const std::string& filename, ArrayBuffer& elements, ElementArray& index, const std::size_t& id = 0);
	static std::vector<MeshPair> ReadOBJ(const std::string& filename);
	static void ReadOBJ(const std::string& filename, std::vector<MeshPair>& input);
};

#endif // OBJ_READER_H