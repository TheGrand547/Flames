#include "Input.h"
#include <glm/gtc/epsilon.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <array>
#include "util.h"
#include "imgui/imgui.h"
#include "imgui/imgui_stdlib.h"
#include "Interpolation.h"
#include "ini.h"

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

	static std::array<float, 6> controllerDeadzones;
	static std::array<int, 6> controllerCurves;
	static std::array<bool, 6> curveSmoothing;
	static bool GamepadActive = true;

	// TODO: Format this
	// TODO: Integrate the ini file stuff

	static std::array<decltype(&Easing::Linear), 7> smoothings
	{
		Easing::Linear, Easing::Quadratic, Easing::EaseOutQuadratic, Easing::Cubic, Easing::EaseOutCubic,
		Easing::Circular, Easing::EaseOutCircular
	};
	static auto easingNames = std::to_array<const char*>({ "Linear", "Quadratic", "Quadratic Ease Out", "Cubic",
		"Cubic Ease Out", "Circular", "Circular Ease Out" });
	static auto buttonNames = std::to_array<const char*>({ "A / Cross", "B / Circle", "X / Square", "Y / Triangle", "Left Bumper", "Right Bumper",
		"Back", "Start", "Guide", "Left Thumbstick", "Right Thumbstick", "D-Pad Up", "D-Pad Right", "D-Pad Down", "D-Pad Left",
		"Left Trigger", "Right Trigger" });
	static auto keyboardInputs = std::to_array<const char*>({ "Input Character", "Shift", "Right Shift", "Control", "Right Control", "Tab",
		"Alt", "Left Alt" });
	static auto buttonInputNames = std::to_array<const char*>({ "Fire Main Weapon", "Autopilot", "Fire Secondary Weapon" });
	static auto stickNames = std::to_array<std::string>({"Left Thumbstick X", "Left Thumbstick Y", "Right Thumbstick X", "Right Thumbstick Y",
		"Right Trigger X", "Right Trigger Y"});

	static std::array<int, buttonInputNames.size()> keyboardBindings, gamepadBindings;

	static glm::vec2 mouseSensitivity{};


	unsigned int IndexFromButton(Gamepad::Button button)
	{
		return std::bit_ceil(static_cast<std::uint16_t>(button));
	}

	Gamepad::Button ButtonFromIndex(unsigned int index)
	{
		return static_cast<Gamepad::Button>(1 << index);
	}

	void Setup() noexcept
	{
		// INI stuff
		mINI::INIFile file("input.ini");
		mINI::INIStructure input;
		if (!file.read(input))
		{
			// No config file, load defaults
			
		}
		// Translate from ini to the configuration
		if (input.has("Gamepad Tuning"))
		{
			for (auto& a : input)
			{
				std::cout << a.first << "L:" << a.second.size() << "\n";
				for (auto& b : a.second)
				{
					std::cout << b.first << "L:" << b.second.size() << "\n";
				}
			}
			auto& local = input["Gamepad Tuning"];
			for (std::size_t i = 0; i < 6; i++)
			{
				// Check for easing functions and such
				const std::string& ref = stickNames[i];
				{
					std::string temp = ref + " Deadzone";
					if (local.has(temp))
					{
						std::string current = local[temp];
						auto result = std::from_chars(current.c_str(), current.c_str() + current.length(), controllerDeadzones[i]);
						std::cout << controllerDeadzones[i] << '\n';
						if (result.ec != std::errc())
						{
							controllerDeadzones[i] = 0.f;
						}
					}
				}
				if (i > 4) // Triggers don't have easing functions
					continue;
				std::string temp = ref + " Easing";
				if (local.has(temp))
				{
					std::string current = local[temp];
					auto result = std::from_chars(current.c_str(), current.c_str() + current.length(), controllerCurves[i]);
					std::cout << controllerCurves[i] << '\n';
					if (result.ec != std::errc())
					{
						controllerCurves[i] = 0;
					}
					controllerCurves[i] = std::clamp(controllerCurves[i], 0, static_cast<int>(easingNames.size()) - 1);
				}
				std::string temp2 = ref + " Sharp Easing";
				if (local.has(temp))
				{
					std::string current = local[temp2];
					int local = 0;
					auto result = std::from_chars(current.c_str(), current.c_str() + current.length(), local);
					std::cout << controllerCurves[i] << '\n';
					if (result.ec != std::errc())
					{
						local = 0;
					}
					curveSmoothing[i] = !!local;
				}
			}
		}
		Input::Gamepad::Setup();
	}
	
	bool Teardown() noexcept
	{
		mINI::INIFile file("input.ini");
		mINI::INIStructure output;

		// TODO: Generate from current data
		for (std::size_t i = 0; i < 6; i++)
		{
			output["Gamepad Tuning"][stickNames[i] + " Deadzone"] = std::format("{:f}", controllerDeadzones[i]);
			if (i < 4)
			{
				output["Gamepad Tuning"][stickNames[i] + " Easing"] = std::format("{}", controllerCurves[i]);
				output["Gamepad Tuning"][stickNames[i] + " Sharp Easing"] = std::format("{}", (curveSmoothing[i]) ? 1 : 0);
			}
		}
		return file.write(output, true);
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
				Gamepad::currentButtons |= (Gamepad::axes[2].x > 0.f) * Gamepad::LeftTrigger;
				Gamepad::currentButtons |= (Gamepad::axes[2].y > 0.f) * Gamepad::RightTrigger;
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

	float deadzone(float in, float threshold)
	{
		if (glm::abs(in) > threshold)
		{
			return in;
		}
		return 0;
	}

	static void ControlTuning(int index)
	{
		ImGui::PushID(123);
		ImGui::SliderFloat("Deadzone", &controllerDeadzones[index], 0.0f, 1.f);
		static bool sharpEdge = false;
		ImGui::Checkbox("Sharp Deadzone", &curveSmoothing[index]);
		ImGui::Combo("Curve", &controllerCurves[index], easingNames.data(), static_cast<int>(easingNames.size()));
		std::array<float, 101> plot{};
		float currentDeadzone = controllerDeadzones[index];

		float duration = 1.f - currentDeadzone;
		for (std::size_t i = 0; i < plot.size(); i++)
		{	
			float delta = i * 1.f / (plot.size() - 1);
			delta = deadzone(delta, currentDeadzone);
			if (!curveSmoothing[index] && delta > currentDeadzone)
			{
				delta = (delta - currentDeadzone) / duration;
			}
			delta = static_cast<float>(smoothings[controllerCurves[index]](delta));
			plot[i] = delta;
		}
		ImGui::PlotLines("Plot", plot.data(), static_cast<int>(plot.size()), 0, nullptr, 0.f, 1.f, ImVec2(0.f, 50.f));
		ImGui::PopID();
	}

	void AxesTuning(int index)
	{
		ImGui::SetNextItemOpen(true, ImGuiCond_Once);
		if (ImGui::TreeNode("X Axis"))
		{
			ControlTuning(2 * index);
			ImGui::TreePop();
		}
		ImGui::SetNextItemOpen(true, ImGuiCond_Once);
		if (ImGui::TreeNode("Y Axis"))
		{
			ControlTuning(2 * index + 1);
			ImGui::TreePop();
		}
	}

	static bool inputConfigurationEnabled = false;

	void ToggleUI()
	{
		inputConfigurationEnabled = !inputConfigurationEnabled;
	}

	void DisplayInput()
	{
		if (GamepadActive)
		{
			ImGui::Begin("Gamepad Status");
			for (int i = 0; i < 3; i++)
			{
				//ImGui::TreePush("###");
				ImGui::Text("%f", Gamepad::Gamepad::CheckAxes(i).x);
				ImGui::SameLine();
				ImGui::Text("%f", Gamepad::Gamepad::CheckAxes(i).y);
			}
			ImGui::End();
		}
	}

	void UIStuff()
	{
		if (!inputConfigurationEnabled)
		{
			return;
		}
		ImGui::Begin("Input Configuration", &inputConfigurationEnabled);	
		if (ImGui::CollapsingHeader("Controller"))
		{
			if (ImGui::TreeNode("Input Tuning"))
			{
				if (ImGui::TreeNode("Left Thumbstick"))
				{
					AxesTuning(0);
					ImGui::TreePop();
				}
				if (ImGui::TreeNode("Right Thumbstick"))
				{
					AxesTuning(1);
					ImGui::TreePop();
				}
				if (ImGui::TreeNode("Bumpers"))
				{
					ImGui::SliderFloat("Left Threshold", &controllerDeadzones[4], -1.f, 1.f);
					ImGui::SetTooltip("-1 is the 'resting' state", ImGui::GetStyle().HoverDelayShort);
					ImGui::SliderFloat("Right Threshold", &controllerDeadzones[5], -1.f, 1.f);
					ImGui::SetTooltip("-1 is the 'resting' state", ImGui::GetStyle().HoverDelayShort);
					ImGui::TreePop();
				}

				ImGui::TreePop();
			}
			if (ImGui::TreeNode("Input Mapping"))
			{
				for (std::size_t i = 0; i < buttonInputNames.size(); i++)
				{
					// TODO: Maybe Left Align all text but I can't be bothered at this point
					ImGui::Text(buttonInputNames[i]); ImGui::SameLine();
					ImGui::PushID(static_cast<int>(i));
					ImGui::Combo("", &gamepadBindings[i], buttonNames.data(), static_cast<int>(buttonNames.size()));
					ImGui::PopID();
				}
				ImGui::TreePop();
			}
		}
		if (ImGui::CollapsingHeader("Mouse + Keyboard"))
		{
			ImGui::SetNextItemOpen(true, ImGuiCond_Once);
			if (ImGui::TreeNode("Keyboard"))
			{
				static char buffer[2];
				for (std::size_t i = 0; i < buttonInputNames.size(); i++)
				{
					ImGui::Combo(buttonInputNames[i], &keyboardBindings[i], keyboardInputs.data(), static_cast<int>(buttonNames.size()));
					if (keyboardBindings[i] == 0)
					{
						ImGui::Spacing();
						ImGui::PushID(static_cast<int>(i));
						ImGui::InputText("Key", buffer, 2);
						ImGui::PopID();
						ImGui::Spacing();
					}
				}
				ImGui::TreePop();
			}
			if (ImGui::TreeNode("Mouse"))
			{
				ImGui::Text("X Sensitivity"); ImGui::SameLine();
				ImGui::SliderFloat("##1", &mouseSensitivity.x, 0.1f, 20.f, "%.3f", ImGuiSliderFlags_Logarithmic);
				ImGui::SetTooltip("Ctrl + Click to Set Manually", ImGui::GetStyle().HoverDelayShort);
				ImGui::Text("Y Sensitivity"); ImGui::SameLine();
				ImGui::SliderFloat("##2", &mouseSensitivity.y, 0.1f, 20.f, "%.3f", ImGuiSliderFlags_Logarithmic);
				ImGui::SetTooltip("Ctrl + Click to Set Manually", ImGui::GetStyle().HoverDelayShort);
				ImGui::TreePop();
			}
		}
		ImGui::End();
	}
};
