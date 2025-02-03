#pragma once
#ifndef INPUT_H
#define INPUT_H
#include <glm/glm.hpp>

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

	// Contains all the relevant player input commands, independent of the keys that actually "fire" them
	struct Keyboard
	{
		glm::vec3 heading;
		bool fireButton;
	};
};


#endif // INPUT_H