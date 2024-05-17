#pragma once
#ifndef USER_INTERFACE_H
#define USER_INFERFACE_H
#include "Button.h"
#include "ScreenRect.h"
#include "Texture2D.h"

/*
Controls the entire non-3d user interface, buttons and all
Formed kind of like so
UserInterface
-Context 1
--Button A
--Button B
--Text Entry 1
-Context 2
--Button C
---Context 3
----
--Button D
...

Hierarchical contexts that can have:
-buttons with arbitrary callbacks
-Scroll widgets(for other widgets)
-Images
-Models

As a proof of concept I will make a main menu with this
*/

class Context 
{
protected:
	std::vector<ButtonBase*> elements{};
public:
	Context() noexcept = default;
	~Context() noexcept;
	void AddButton(ButtonBase* button);
	void Clear() noexcept;
	void Update(MouseStatus& status);

	void Draw();

	template<class T, typename... Args> T* AddButton(Args&&... args)
	requires requires {
		std::is_base_of<ButtonBase, T>();
	}
	{
		T* temp = new T(std::forward<Args>(args)...);
		if (temp)
		{
			this->elements.push_back(static_cast<ButtonBase*>(temp));
		}
		return temp;
	}
};


class UserInterface
{
protected:

public:
	enum EventType
	{
		ScrollUp, ScrollDown, ButtonPress, ButtonHold, ButtonRelease
	};

	void ApplyEvent(EventType event); // TODO: Maybe a struct with extra data? idk


};

#endif // USER_INTERFACE_H
