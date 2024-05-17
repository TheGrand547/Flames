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
	glm::vec2 position;
	unsigned char buttons;

	constexpr bool CheckButton(MouseButton button) const noexcept { return this->buttons & button; }
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
	inline virtual bool MouseUpdate(const MouseStatus& status) noexcept { return false; }
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
	inline virtual bool MouseUpdate(const MouseStatus& status) noexcept override
	{
		this->hovered = this->baseRect.Contains(status.position);
		if (this->hovered && status.CheckButton(this->trigger))
		{
			this->callback(this->id);
		}
		return false;
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
		fixed = glm::max(fixed, { this->baseRect.z, this->baseRect.w });
		this->baseRect.z = fixed.x;
		this->baseRect.w = fixed.y;
	}
};

typedef Button<TrivialCallback> BasicButton;

#endif // BUTTON_H
