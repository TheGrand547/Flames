#include "Input.h"
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
	int Gamepad::currentGamepad;
	bool Gamepad::Active;

	static bool GamepadActive = true;

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

	void Gamepad::Setup() noexcept
	{
		Gamepad::currentGamepad = -1;
		Input::GamepadActive = false;
		for (int i = 0; i < GLFW_JOYSTICK_LAST; i++)
		{
			if (glfwJoystickIsGamepad(GLFW_JOYSTICK_1 + i))
			{
				Gamepad::currentGamepad = i;
				Input::GamepadActive = false;
				break;
			}
		}
	}

	void Gamepad::ControllerStatusCallback(int joystick, int event)
	{
		if (event == GLFW_CONNECTED)
		{
			// Pick one
			std::cout << "YIppie!" << joystick << '\n';
			Gamepad::currentGamepad = joystick;
		}
		else if (event == GLFW_DISCONNECTED)
		{
			// Get rid of current controller, and set the current one to the one that was last plugged in, at least I assume
			// that's what the order is all about
			Gamepad::currentGamepad = -1; // Invalid joystick
			for (int i = GLFW_JOYSTICK_LAST; i >= 0; i--)
			{
				int exists = glfwJoystickIsGamepad(GLFW_JOYSTICK_1 + i);
				currentGamepad = i;
				break;
			}
		}
	}

	void Gamepad::Deactivate()
	{
		GamepadActive = false;
	}



	bool ControllerActive()
	{
		return GamepadActive;
	}

	// TODO: Official, non ad-hoc, binding
	Keyboard UpdateStatus()
	{
		if (GamepadActive)
		{
			// TODO: Return to this input
			return Keyboard();
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
				}
				Gamepad::axes[0] = glm::vec2(input.axes[0], input.axes[1]);
				Gamepad::axes[1] = glm::vec2(input.axes[2], input.axes[3]);
				Gamepad::axes[2] = glm::vec2(input.axes[4], input.axes[5]);
				Gamepad::risingEdge  =  Gamepad::currentButtons & (~Gamepad::oldButtons);
				Gamepad::fallingEdge = ~Gamepad::currentButtons & ( Gamepad::oldButtons);
				Gamepad::oldButtons  =  Gamepad::currentButtons;
				GamepadActive |= !!Gamepad::currentButtons;
			}
			else
			{
				// Invalid gamepad for some reason, zero out all info
				Gamepad::oldButtons = 0;
				Gamepad::currentButtons = 0;
				Gamepad::risingEdge = 0;
				Gamepad::fallingEdge = 0;
				Gamepad::axes[0] = glm::vec2(0.f);
				Gamepad::axes[1] = glm::vec2(0.f);
				Gamepad::axes[2] = glm::vec2(0.f);
				GamepadActive = false;
			}
		}
		else
		{
			GamepadActive = false;
		}
	}
};
