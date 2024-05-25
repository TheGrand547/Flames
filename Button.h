#pragma once
#ifndef BUTTON_H
#define BUTTON_H
#include <array>
#include "Font.h"
#include "ScreenRect.h"
#include "Texture2D.h"

typedef void (*TrivialCallback)(std::size_t id);

struct Mouse
{
protected:
	static unsigned char oldButtons;
	static unsigned char buttons;
	static unsigned char risingEdge;
	static unsigned char fallingEdge;
	static glm::vec2 position;
public:
	enum Button : unsigned char
	{
		Button1 = 1 << 0,
		Button2 = 1 << 1,
		Button3 = 1 << 2,
		Button4 = 1 << 3,
		Button5 = 1 << 4,
		Button6 = 1 << 5,
		Button7 = 1 << 6,
		Button8 = 1 << 7,
		ButtonLeft = Button1,
		ButtonRight = Button2,
		ButtonMiddle = Button3,
	};

	static inline constexpr bool CheckButton(Mouse::Button button) noexcept
	{
		return Mouse::buttons & button;
	}
	static inline constexpr bool CheckRising(Mouse::Button button) noexcept
	{
		return Mouse::risingEdge & button;
	}
	static inline constexpr bool CheckFalling(Mouse::Button button) noexcept
	{
		return Mouse::fallingEdge & button;
	}
	static inline constexpr glm::vec2 GetPosition() noexcept
	{
		return Mouse::position;
	}
	static inline constexpr void SetButton(Mouse::Button button, bool flag) noexcept
	{
		Mouse::buttons = (Mouse::buttons & ~(1 << static_cast<unsigned char>(button))) | (flag << static_cast<unsigned char>(button));
	};
	static inline constexpr void SetButton(int button, bool flag) noexcept
	{
		Mouse::buttons = (Mouse::buttons & ~(1 << static_cast<unsigned char>(button))) | (flag << static_cast<unsigned char>(button));
	};
	static inline constexpr void SetPosition(const glm::vec2& pos) noexcept
	{
		Mouse::position = pos;
	};
	static inline constexpr void SetPosition(const float& x, const float& y) noexcept
	{
		Mouse::position.x = x;
		Mouse::position.y = y;
	};
	static inline constexpr void UpdateEdges() noexcept
	{
		Mouse::risingEdge = Mouse::buttons & (~Mouse::oldButtons);
		Mouse::fallingEdge = ~Mouse::buttons & (Mouse::oldButtons);
		Mouse::oldButtons = Mouse::buttons;
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
	inline virtual void MouseUpdate() noexcept {}
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
	Mouse::Button trigger;
	Texture2D alternateTexture;
public:
	const std::size_t id;

	inline Button(ScreenRect rect, Callback callback, Mouse::Button trigger = Mouse::ButtonLeft, std::size_t id = 0)
		: ButtonBase(rect), hovered(false), callback(callback), trigger(trigger), id((id) ? id : std::bit_cast<std::size_t>(this)) {}
	inline virtual ~Button() noexcept {}
	inline virtual void MouseUpdate() noexcept override
	{
		this->hovered = this->baseRect.Contains(Mouse::GetPosition());
		if (this->hovered && Mouse::CheckRising(this->trigger))
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
