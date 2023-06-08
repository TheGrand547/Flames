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

std::size_t Buffer::Size() const
{
	return this->length;
}

void Buffer::BindBuffer() const
{
	glBindBuffer((GLenum) this->bufferType, this->buffer);
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

void Buffer::Generate(BufferType type, BufferAccess access, GLsizeiptr size)
{
	this->CleanUp();
	this->bufferType = type;
	glGenBuffers(1, &this->buffer);
	if (size)
	{
		this->Reserve(access, size);
	}
}

void Buffer::Reserve(BufferAccess access, GLsizeiptr size)
{
	if (this->buffer)
	{
		glNamedBufferData(this->buffer, size, nullptr, (GLenum) access);
		this->length = size;
	}
}
