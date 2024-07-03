#pragma once
#ifndef OBJ_READER_H
#define OBJ_READER_H
#include <string>
#include "Buffer.h"


struct OBJReader
{
	static void ReadOBJ(const std::string& filename, ArrayBuffer& elements, ElementArray& index);
};

#endif // OBJ_READER_H