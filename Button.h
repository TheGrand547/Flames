#pragma once
#ifndef BUTTON_H
#define BUTTON_H
#include <array>
//#include <GLFW/glfw3.h>
#include "ScreenRect.h"

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
	// True -> visual state updated, false -> no visual state update
	virtual bool MouseUpdate(const MouseStatus& status) { return false; }
	virtual ~ButtonBase() {}
};

template<typename Callback>
class Button : public ButtonBase
{
protected:
	ScreenRect rect;
	Callback callback;
	MouseButton trigger;
public:
	const std::size_t id;

	Button(ScreenRect rect, Callback callback, MouseButton trigger = MouseButtonLeft, std::size_t id = 0)
		: rect(rect), callback(callback), trigger(trigger), id((id) ? id : std::bit_cast<std::size_t>(this)) {}
	virtual ~Button() {}
	inline virtual bool MouseUpdate(const MouseStatus& status) override
	{
		if (status.CheckButton(this->trigger) && this->rect.Contains(status.position))
		{
			this->callback(this->id);
		}
		return false;
	}
};

typedef Button<TrivialCallback> BasicButton;

/*
template<typename T> class ButtonCallback : public ButtonBase
{

};*/

#endif // BUTTON_H
