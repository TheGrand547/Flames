#include "UniformBuffer.h"

UniformBuffer::UniformBuffer() : Buffer(), bufferBinding(0)
{

}

UniformBuffer::~UniformBuffer()
{
	this->CleanUp();
}