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
