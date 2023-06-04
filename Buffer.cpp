#include "Buffer.h"

Buffer::Buffer() : buffer(0), bufferType(ArrayBuffer), length(0) 
{

}

Buffer::Buffer(BufferType type) : buffer(0), bufferType(type), length(0)
{
	this->Generate(type);
}

Buffer::Buffer(Buffer&& other) noexcept
{
	this->CleanUp();
	this->buffer = other.buffer;
	this->bufferType = other.bufferType;
	other.CleanUp();
}

Buffer::~Buffer()
{
	this->CleanUp();
}

size_t Buffer::Size() const
{
	return this->length;
}

void Buffer::BindBuffer() const
{
	if (this->buffer)
	{
		glBindBuffer(this->bufferType, this->buffer);
	}
}

void Buffer::CleanUp()
{
	if (this->buffer)
	{
		glDeleteBuffers(1, &this->buffer);
	}
	this->buffer = 0;
	this->length = 0;
}

void Buffer::Generate(BufferType type, GLsizeiptr size)
{
	this->CleanUp();
	this->bufferType = type;
	glGenBuffers(1, &this->buffer);
	if (size)
	{
		this->Reserve(size);
	}
}

void Buffer::Reserve(GLsizeiptr size) const
{
	if (this->buffer)
	{
		glNamedBufferData(this->buffer, size, NULL, this->bufferType);
	}
}
