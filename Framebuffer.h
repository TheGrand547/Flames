#pragma once
#ifndef FRAME_BUFFER_H
#define FRAME_BUFFER_H
#include <type_traits>
#include <variant>
#include <vector>
#include "log.h"
#include "Texture2D.h"

constexpr int MaxFrameBufferColorAttachments = 16;


enum FrameBufferAttachments
{
	OnlyColor,      // No Depth or stencil(might be invalid?)
	Depth,          // Only depth texture
	Stencil,        // Only stencil texture
	DepthStencil,   // Combined Depth and Stencil texture
	DepthAndStencil // Depth and Stencil as seperate textures
};

enum FrameBufferTypes
{
	DrawBuffer      = GL_DRAW_FRAMEBUFFER,
	ReadBuffer      = GL_READ_FRAMEBUFFER,
	ReadWriteBuffer = GL_FRAMEBUFFER,
};

// TODO: One of these but for renderbuffers instead, faster

template<std::size_t ColorAttachments = 1, FrameBufferAttachments buffers = Depth>
	requires requires
{
	ColorAttachments >= 0 && ColorAttachments < 16; // TODO: Get from the GL library thing
}
class Framebuffer
{
protected:
	static inline constexpr bool HasDepth    = (buffers == Depth) || (buffers == DepthAndStencil);
	static inline constexpr bool HasStencil  = (buffers == Stencil) || (buffers == DepthAndStencil);
	static inline constexpr bool HasCombined = buffers == DepthStencil;
	static inline constexpr bool HasColor    = ColorAttachments > 0;
	static inline constexpr bool SingleColor = ColorAttachments == 1;
	static inline constexpr bool MultiColor  = ColorAttachments > 1;

	std::conditional_t<HasDepth, Texture2D, std::monostate> depth;
	std::conditional_t<HasStencil, Texture2D, std::monostate> stencil;
	std::conditional_t<HasCombined, Texture2D, std::monostate> depthStencil;
	std::conditional_t<HasColor, std::array<Texture2D, ColorAttachments>, std::monostate> colorBuffers{};

	static constexpr std::conditional_t<HasColor, std::array<GLenum, ColorAttachments>, std::monostate> drawBuffermacro =
		[]() {
		std::array<GLenum, ColorAttachments> temp{};
		temp.fill(GL_COLOR_ATTACHMENT0);
		for (int i = 0; i < ColorAttachments; i++)
		{
			temp[i] += i;
		}
		return temp;
		}();

	GLuint frameBuffer = 0;
public:
	~Framebuffer()
	{
		this->CleanUp();
	}

	void CleanUp()
	{
		if (this->frameBuffer)
		{
			glDeleteFramebuffers(1, &this->frameBuffer);
			this->frameBuffer = 0;
		}
		if constexpr (HasColor)
		{
			for (auto& element : this->colorBuffers)
			{
				element.CleanUp();
			}
		}
		if constexpr (HasDepth)
		{
			this->depth.CleanUp();
		}
		if constexpr(HasStencil)
		{
			this->stencil.CleanUp();
		}
		if constexpr (HasCombined)
		{
			this->depthStencil.CleanUp();
		}
	}

	template<typename = std::enable_if_t<HasDepth>> Texture2D& GetDepth()
	{
		return this->depth;
	}

	template<typename = std::enable_if_t<HasStencil>> Texture2D& GetStencil()
	{
		return this->stencil;
	}

	template<typename = std::enable_if_t<HasCombined>> Texture2D& GetDepthStencil()
	{
		return this->depthStencil;
	}

	template<typename = std::enable_if_t<SingleColor>> Texture2D& GetColor()
	{
		return this->colorBuffers[0];
	}

	template<typename = std::enable_if_t<HasColor>> Texture2D& GetColorBuffer(const std::size_t i = 0)
	{
		assert(i < ColorAttachments);
		return this->colorBuffers[i];
	}

	template<std::size_t i, typename = std::enable_if_t<HasColor>> Texture2D& GetColorBuffer()
	requires requires {
		0 <= i && i < ColorAttachments;
	}
	{
		return this->colorBuffers[i];
	}

	bool Assemble()
	{
		if (this->frameBuffer)
		{
			glDeleteFramebuffers(1, &this->frameBuffer);
			this->frameBuffer = 0;
		}
		glGenFramebuffers(1, &this->frameBuffer);
		glBindFramebuffer(GL_FRAMEBUFFER, this->frameBuffer);
		if constexpr (HasColor)
		{
			for (std::size_t i = 0; i < ColorAttachments; i++)
			{
				glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + (GLenum) i, GL_TEXTURE_2D, this->colorBuffers[i].GetGLTexture(), 0);
			}
		}
		if constexpr (HasDepth)
		{
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, this->depth.GetGLTexture(), 0);
		}
		if constexpr (HasStencil)
		{
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_STENCIL_ATTACHMENT, GL_TEXTURE_2D, this->stencil.GetGLTexture(), 0);
		}
		if constexpr (HasCombined)
		{
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, this->depthStencil.GetGLTexture(), 0);
		}
		GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
		if (status != GL_FRAMEBUFFER_COMPLETE)
		{
			switch (status)
			{
			case GL_FRAMEBUFFER_UNDEFINED: Log("Default framebuffer unspecified."); break;
			case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT: Log("At least one attachment point is incomplete."); break;
			case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT: Log("No images attached to the framebuffer."); break;
			case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER: Log("Confused, have the enum name 'GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER'"); break;
			case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER: Log("Confused, have the enum name 'GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER'"); break;
			case GL_FRAMEBUFFER_UNSUPPORTED: Log("Incompatible internal formats of attachments."); break;
			case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE: Log("RenderBuffer samples do not agree for all attachments"); break;
			case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS: Log("More confusing ones again 'GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS'"); break;
			default: Log("Some framebuffer error that isn't one of the given, dunno what to do with that."); break;
			}
		}
		return glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
	}

	inline void Bind()
	{
		glBindFramebuffer(GL_FRAMEBUFFER, this->frameBuffer);
		if constexpr (HasColor)
		{
			glDrawBuffers(ColorAttachments, drawBuffermacro.data());
		}
	}
};

typedef Framebuffer<1, OnlyColor> ColorFrameBuffer;


inline void BindDefaultFrameBuffer()
{
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}
#endif // FRAME_BUFFER_H
