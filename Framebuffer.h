#pragma once
#ifndef FRAME_BUFFER_H
#define FRAME_BUFFER_H
#include <type_traits>
#include <variant>
#include <vector>
#include "log.h"
#include "Texture2D.h"

// Absolute hack move, but I can't really find it out at compile time soooooo
constexpr int MaxFrameBufferColorAttachments = 16;

enum FrameBufferAttachments
{
	OnlyColor,      // No Depth or stencil(might be invalid?)
	Depth,          // Only depth texture
	Stencil,        // Only stencil texture
	DepthStencil,   // Combined Depth and Stencil texture
	DepthAndStencil // Depth and Stencil as seperate textures
};

// TODO: One of these but with renderbuffers instead, faster

template<std::size_t ColorAttachments = 1, FrameBufferAttachments buffers = Depth>
	requires requires
{
	ColorAttachments >= 0 && ColorAttachments < MaxFrameBufferColorAttachments;
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

	[[no_unique_address]] std::conditional_t<HasDepth, Texture2D, std::monostate> depth;
	[[no_unique_address]] std::conditional_t<HasStencil, Texture2D, std::monostate> stencil;
	[[no_unique_address]] std::conditional_t<HasCombined, Texture2D, std::monostate> depthStencil;
	[[no_unique_address]] std::conditional_t<HasColor, std::array<Texture2D, ColorAttachments>, std::monostate> colorBuffers{};

	static constexpr std::conditional_t<HasColor, std::array<GLenum, ColorAttachments>, std::monostate> drawBufferMacro =
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
	Framebuffer() = default;

	Framebuffer(Framebuffer<ColorAttachments, buffers>&& other) noexcept
	{
		if constexpr (HasDepth) std::swap(this->depth, other.depth);
		if constexpr (HasStencil) std::swap(this->stencil, other.stencil);
		if constexpr (HasCombined) std::swap(this->depthStencil, other.depthStencil);
		if constexpr (HasColor) this->colorBuffers = std::move(other.colorBuffers);
		std::swap(this->frameBuffer, other.frameBuffer);
	}

	~Framebuffer()
	{
		this->CleanUp();
	}

	Framebuffer<ColorAttachments, buffers>& operator=(Framebuffer<ColorAttachments, buffers>&& other) noexcept
	{
		if constexpr (HasDepth)    std::swap(this->depth, other.depth);
		if constexpr (HasStencil)  std::swap(this->stencil, other.stencil);
		if constexpr (HasCombined) std::swap(this->depthStencil, other.depthStencil);
		if constexpr (HasColor)    this->colorBuffers = std::move(other.colorBuffers);
		std::swap(this->frameBuffer, other.frameBuffer);
		return *this;
	}

	inline GLuint GetFrameBuffer() const noexcept
	{
		return this->frameBuffer;
	}

	void CleanFramebuffer()
	{
		if (this->frameBuffer)
		{
			glDeleteFramebuffers(1, &this->frameBuffer);
			this->frameBuffer = 0;
		}
	}

	void CleanUp()
	{
		this->CleanFramebuffer();
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

	// TODO: version of assemble where it creates the textures at a specified size and format

	bool Assembled() const noexcept
	{
		return this->frameBuffer != 0;
	}

	bool Assemble()
	{
		this->CleanFramebuffer();
		glGenFramebuffers(1, &this->frameBuffer);
		glBindFramebuffer(GL_FRAMEBUFFER, this->frameBuffer);
		if constexpr (HasColor)
		{
			for (std::size_t i = 0; i < ColorAttachments; i++)
			{
				glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + static_cast<GLenum>(i), GL_TEXTURE_2D, this->colorBuffers[i].GetGLTexture(), 0);
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
		return status == GL_FRAMEBUFFER_COMPLETE;
	}

	bool Create(std::size_t width, std::size_t height, TextureFormatInternal internal = InternalRGBA)
	{
		if constexpr (HasColor)
		{
			for (std::size_t i = 0; i < ColorAttachments; i++)
			{
				this->colorBuffers[i].CreateEmptyWithFilters(width, height, internal);
			}
		}
		if constexpr (HasDepth)
		{
			this->depth.CreateEmptyWithFilters(width, height, InternalDepth32);
		}
		if constexpr (HasStencil)
		{
			this->stencil.CreateEmptyWithFilters(width, height, InternalStencil);
		}
		if constexpr (HasCombined)
		{
			this->depthStencil.CeateEmptyWithFiltes(width, height, InternalDepthStencil);
		}
		return this->Assemble();
	}

	// Binds the framebuffer to both the read and write positions
	inline void Bind()
	{
		glBindFramebuffer(GL_FRAMEBUFFER, this->frameBuffer);
		if constexpr (HasColor)
		{
			glDrawBuffers(ColorAttachments, drawBufferMacro.data());
		}
		if constexpr (SingleColor)
		{
			glViewport(0, 0, this->GetColor().GetWidth(), this->GetColor().GetHeight());
		}
	}

	inline void BindDraw()
	{
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, this->frameBuffer);
	}

	inline void BindRead()
	{
		glBindFramebuffer(GL_READ_FRAMEBUFFER, this->frameBuffer);
	}

	template<typename = std::enable_if_t<HasColor>>
	inline void ReadColorIntoTexture(const std::size_t i = 0)
	{
		assert(i < ColorAttachments);
		glBindFramebuffer(GL_READ_FRAMEBUFFER, this->frameBuffer);
		glReadBuffer(GL_COLOR_ATTACHMENT0 + static_cast<GLenum>(i));
	}

	// TODO: incorporate the other variables
	template<typename = std::enable_if_t<HasColor>>
	inline void ReadColorIntoTexture(Texture2D& texture, const std::size_t i = 0)
	{
		this->ReadColorIntoTexture(i);
		Texture2D& source = this->GetColorBuffer(i);
		texture.CopyFromFramebuffer(source.GetSize());

	}

	// TODO: Framebuffer blits and reads but I don't wanna do those so who cares
};

typedef Framebuffer<1, OnlyColor> ColorFrameBuffer;

#endif // FRAME_BUFFER_H
