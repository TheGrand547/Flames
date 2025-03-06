#pragma once
#ifndef INPUT_H
#define INPUT_H
#include <glm/glm.hpp>
#include <glew.h>
#include <GLFW/glfw3.h>
#include <array>

namespace Input
{
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

	struct Gamepad
	{
	protected:
		static std::uint16_t oldButtons;
		static std::uint16_t currentButtons;
		static std::uint16_t risingEdge;
		static std::uint16_t fallingEdge;
		static std::array<glm::vec2, 3> axes;
		static bool Active;

		static int currentGamepad;
	public:
		enum Button
		{
			A               = 1 << GLFW_GAMEPAD_BUTTON_A,
			B               = 1 << GLFW_GAMEPAD_BUTTON_B,
			X               = 1 << GLFW_GAMEPAD_BUTTON_X,
			Y               = 1 << GLFW_GAMEPAD_BUTTON_Y,
			LeftBumper      = 1 << GLFW_GAMEPAD_BUTTON_LEFT_BUMPER,
			RightBumper     = 1 << GLFW_GAMEPAD_BUTTON_RIGHT_BUMPER,
			BackButton      = 1 << GLFW_GAMEPAD_BUTTON_BACK,
			Start           = 1 << GLFW_GAMEPAD_BUTTON_START,
			Guide           = 1 << GLFW_GAMEPAD_BUTTON_GUIDE,
			LeftThumbstick  = 1 << GLFW_GAMEPAD_BUTTON_LEFT_THUMB,
			RightThumbstick = 1 << GLFW_GAMEPAD_BUTTON_RIGHT_THUMB,
			DPadUp          = 1 << GLFW_GAMEPAD_BUTTON_DPAD_UP,
			DPadRight       = 1 << GLFW_GAMEPAD_BUTTON_DPAD_RIGHT,
			DPadDown        = 1 << GLFW_GAMEPAD_BUTTON_DPAD_DOWN,
			DPadLeft        = 1 << GLFW_GAMEPAD_BUTTON_DPAD_LEFT,
			LeftTrigger     = 1 << (GLFW_GAMEPAD_BUTTON_LAST + 1),
			RightTrigger    = 1 << (GLFW_GAMEPAD_BUTTON_LAST + 2),
		};

		static inline constexpr bool CheckButton(Gamepad::Button button)
		{
			return Gamepad::currentButtons & button;
		}
		static inline constexpr bool CheckRisng(Gamepad::Button button)
		{
			return Gamepad::risingEdge & button;
		}
		static inline constexpr bool CheckFalling(Gamepad::Button button)
		{
			return Gamepad::fallingEdge & button;
		}
		
		static inline glm::vec2 CheckAxes(int index) // TODO: You know the clearer stuff yknow
		{
			return Gamepad::axes[index];
		}

		// Only to be called once per update loop, otherwise edges will be garbage
		static void Update();
		static void Setup() noexcept;
		static void ControllerStatusCallback(int joystick, int event);

		static void Deactivate();
	};

	// Contains all the relevant player input commands, independent of the keys that actually "fire" them
	struct Keyboard
	{
		glm::vec4 heading;
		bool fireButton;
		bool popcornFire;
		bool cruiseControl;
	};

	void ControllerStuff();
	
	bool ControllerActive();
	Keyboard UpdateStatus();
};


#endif // INPUT_H