#include "UniformBuffer.h"

UniformBuffer::UniformBuffer() : Buffer(), bufferBinding(0)
{
	this->bufferType = UniformBufferObject;
}

UniformBuffer::~UniformBuffer()
{
	this->CleanUp();
}