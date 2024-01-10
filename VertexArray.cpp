#include "VertexArray.h"

VertexArray::VertexArray(GLuint array, GLsizei stride) : array(array), stride(stride), strides()
{

}

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
