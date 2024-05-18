#pragma once
#ifndef BUTTON_H
#define BUTTON_H
#include <array>
#include "Font.h"
#include "ScreenRect.h"
#include "Texture2D.h"

typedef void (*TrivialCallback)(std::size_t id);

enum MouseButton : unsigned char
{
	MouseButton1 = 1 << 0,
	MouseButton2 = 1 << 1,
	MouseButton3 = 1 << 2,
	MouseButton4 = 1 << 3,
	MouseButton5 = 1 << 4,
	MouseButton6 = 1 << 5,
	MouseButton7 = 1 << 6,
	MouseButton8 = 1 << 7,
	MouseButtonLeft = MouseButton1,
	MouseButtonRight = MouseButton2,
	MouseButtonMiddle = MouseButton3,
};



struct MouseStatus
{
protected:
	unsigned char oldButtons;
	glm::vec2 position;
	unsigned char buttons;
	unsigned char risingEdge;
	unsigned char fallingEdge;
public:
	inline constexpr bool CheckButton(MouseButton button) const noexcept
	{
		return this->buttons & button;
	}
	inline constexpr bool CheckRising(MouseButton button) const noexcept
	{
		return this->risingEdge & button;
	}
	inline constexpr bool CheckFalling(MouseButton button) const noexcept
	{
		return this->fallingEdge & button;
	}
	inline constexpr glm::vec2 GetPosition() const noexcept
	{
		return this->position;
	}
	inline constexpr void SetButton(MouseButton button, bool flag) noexcept
	{
		this->buttons = (this->buttons & ~(1 << static_cast<unsigned char>(button))) | (flag << static_cast<unsigned char>(button));
	};
	inline constexpr void SetButton(int button, bool flag) noexcept
	{
		this->buttons = (this->buttons & ~(1 << static_cast<unsigned char>(button))) | (flag << static_cast<unsigned char>(button));
	};
	inline constexpr void SetPosition(const glm::vec2& pos) noexcept 
	{
		this->position = pos;
	};
	inline constexpr void SetPosition(const float& x, const float& y) noexcept 
	{
		this->position.x = x; 
		this->position.y = y;
	};
	inline constexpr void UpdateEdges() noexcept
	{
		this->risingEdge = this->buttons & (~this->oldButtons);
		this->fallingEdge = ~this->buttons & (this->oldButtons);
		this->oldButtons = this->buttons;
	}
};

// Abstract interface
struct ButtonBase
{
protected:
	ScreenRect baseRect;
	Texture2D baseTexture;
public:
	inline ButtonBase(ScreenRect rect = glm::vec4(0, 0, 1, 1)) noexcept : baseRect(rect) {}
	// True -> visual state updated, false -> no visual state update
	inline virtual void MouseUpdate(const MouseStatus& status) noexcept {}
	virtual const Texture2D& GetTexture() const noexcept;
	virtual ScreenRect GetRect() const noexcept;
	inline virtual ~ButtonBase() noexcept {}
};

template<typename Callback>
class Button : public ButtonBase
{
protected:
	bool hovered;
	Callback callback;
	MouseButton trigger;
	Texture2D alternateTexture;
public:
	const std::size_t id;

	inline Button(ScreenRect rect, Callback callback, MouseButton trigger = MouseButtonLeft, std::size_t id = 0)
		: ButtonBase(rect), hovered(false), callback(callback), trigger(trigger), id((id) ? id : std::bit_cast<std::size_t>(this)) {}
	inline virtual ~Button() noexcept {}
	inline virtual void MouseUpdate(const MouseStatus& status) noexcept override
	{
		this->hovered = this->baseRect.Contains(status.GetPosition());
		if (this->hovered && status.CheckRising(this->trigger))
		{
			this->callback(this->id);
		}
	}

	virtual const Texture2D& GetTexture() const noexcept override
	{
		return (this->hovered) ? this->alternateTexture : this->baseTexture;
	}

	void SetMessages(const std::string& off, const std::string& on, ASCIIFont& font)
	{
		font.RenderToTexture(this->baseTexture, off);
		font.RenderToTexture(this->alternateTexture, on, glm::vec4(1, 0, 0, 1));
		glm::vec2 fixed = glm::max(this->baseTexture.GetSize(), this->alternateTexture.GetSize());
		fixed = glm::min(fixed, { this->baseRect.z, this->baseRect.w });
		this->baseRect.z = fixed.x;
		this->baseRect.w = fixed.y;
	}
};

typedef Button<TrivialCallback> BasicButton;

#endif // BUTTON_H
