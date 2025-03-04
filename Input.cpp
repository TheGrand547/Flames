#include "Input.h"
#include <GLFW/glfw3.h>
#include <glm/gtc/epsilon.hpp>
#include <iostream>
#include <array>
#include "util.h"

namespace Input
{
	unsigned char Mouse::oldButtons;
	unsigned char Mouse::buttons;
	unsigned char Mouse::risingEdge;
	unsigned char Mouse::fallingEdge;
	glm::vec2 Mouse::position;

	std::uint16_t Gamepad::oldButtons;
	std::uint16_t Gamepad::currentButtons;
	std::uint16_t Gamepad::risingEdge;
	std::uint16_t Gamepad::fallingEdge;
	std::array<glm::vec2, 3> Gamepad::axes;

	struct GamepadState
	{
		std::uint16_t currentButtons;
		std::uint16_t previous;
		std::array<glm::vec2, 3> axes;
	};

	static int currentGamepad = -1;
	static bool keyboardActive = true;
	//static GLFWgamepadstate currentState, transitions, risingEdge, fallingEdge;

	void ControllerStuff()
	{
		for (int i = 0; i < GLFW_JOYSTICK_LAST; i++)
		{
			auto exists = glfwJoystickIsGamepad(GLFW_JOYSTICK_1 + i);
			std::cout << std::boolalpha << "i: " << (exists == GLFW_TRUE);
			if (exists)
			{
				std::cout << ": " << glfwGetGamepadName(GLFW_JOYSTICK_1 + i) << '\n';
			}
			std::cout << "\n";
		}
	}

	void ControllerStatusCallback(int joystick, int event)
	{
		if (event == GLFW_CONNECTED)
		{
			// Pick one
			std::cout << "YIppie!" << joystick << '\n';
			currentGamepad = joystick;
		}
		else if (event == GLFW_DISCONNECTED)
		{
			// Get rid of current controller, and set the current one to the one that was last plugged in, at least I assume
			// that's what the order is all about
			currentGamepad = -1; // Invalid joystick
			for (int i = GLFW_JOYSTICK_LAST; i >= 0; i--)
			{
				int exists = glfwJoystickIsGamepad(GLFW_JOYSTICK_1 + i);
				currentGamepad = i;
				break;
			}
		}
	}

	/*
	GLFWgamepadstate GetGamepadState()
	{
		return currentState;
	}
	GLFWgamepadstate GetGamepadStateRising()
	{
		return risingEdge;
	}
	GLFWgamepadstate GetGamepadStateFalling()
	{
		return fallingEdge;
	}
	GLFWgamepadstate GetGamepadStateDifference()
	{
		return transitions;
	}
	*/
	Keyboard UpdateStatus()
	{
		if (keyboardActive)
		{
			// TODO: Return to this inpu
		}
		else
		{
			return Keyboard();
		}
	}

	void Gamepad::Update()
	{
		if (currentGamepad != -1)
		{
			GLFWgamepadstate input;
			if (glfwGetGamepadState(GLFW_JOYSTICK_1 + currentGamepad, &input))
			{
				Gamepad::oldButtons = Gamepad::currentButtons;
				Gamepad::currentButtons = 0;
				// Update the relevant falling/rising edges
				for (int i = 0; i < GLFW_GAMEPAD_BUTTON_LAST; i++)
				{
					Gamepad::currentButtons |= (input.buttons[i] == GLFW_PRESS) << i;
					/*
					transitions.buttons[i] = (input.buttons[i] == GLFW_PRESS) != (currentState.buttons[i] == GLFW_PRESS);
					// Old unpressed, new pressed
					risingEdge.buttons[i]  = (input.buttons[i] == GLFW_PRESS) && (currentState.buttons[i] != GLFW_PRESS);
					// Old pressed, new released
					fallingEdge.buttons[i] = (input.buttons[i] != GLFW_PRESS) && (currentState.buttons[i] == GLFW_PRESS);
					*/
				}
				for (int i = 0; i < GLFW_GAMEPAD_AXIS_LAST; i += 2)
				{
					Gamepad::axes[i] = glm::vec2(input.axes[i], input.axes[i + 1]);

					/*
					transitions.axes[i] = glm::epsilonNotEqual(input.axes[i], currentState.axes[i], EPSILON);

					// Can't think of any use for these, but good to ghave them I guess
					risingEdge.axes[i] = input.axes[i] - currentState.axes[i];
					fallingEdge.axes[i] = currentState.axes[i] - input.axes[i];
					*/
				}
				//currentState = input;
				Gamepad::risingEdge  =  Gamepad::currentButtons & (~Gamepad::oldButtons);
				Gamepad::fallingEdge = ~Gamepad::currentButtons & ( Gamepad::oldButtons);
				Gamepad::oldButtons  =  Gamepad::currentButtons;
			}
			else
			{
				// Invalid gamepad for some reason
				Gamepad::oldButtons = 0;
				Gamepad::currentButtons = 0;
				Gamepad::risingEdge = 0;
				Gamepad::fallingEdge = 0;
				keyboardActive = true;
			}
		}
		else
		{
			keyboardActive = true;
		}
	}
};
