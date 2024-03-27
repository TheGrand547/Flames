#pragma once
#ifndef BUTTON_H
#define BUTTON_H
#include <array>
//#include <GLFW/glfw3.h>
#include "ScreenRect.h"

struct MouseStatus
{
	glm::vec2 position;
	unsigned char buttons;
};

// Abstract interface
struct ButtonBase
{
	virtual void MouseUpdate(const MouseStatus& status) {}
};


class Button : public ButtonBase
{
protected:
	ScreenRect rect;

};

template<typename T> class ButtonCallback : public ButtonBase
{

};

#endif // BUTTON_H
