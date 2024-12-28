#pragma once
#ifndef OBJ_READER_H
#define OBJ_READER_H
#include <string>
#include "Buffer.h"


struct OBJReader
{
	static void ReadOBJ(const std::string& filename, ArrayBuffer& elements, ElementArray& index, const std::size_t& id = 0);
};

#endif // OBJ_READER_H