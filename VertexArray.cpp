#include "VertexArray.h"

VertexArray::~VertexArray()
{
	this->CleanUp();
}

void VertexArray::CleanUp()
{
	if (this->array)
	{
		glDeleteVertexArrays(1, &this->array);
		this->array = 0;
	}
	this->stride = 0;
}
