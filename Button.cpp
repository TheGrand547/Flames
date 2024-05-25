#include "Button.h"

ScreenRect ButtonBase::GetRect() const noexcept
{
	return this->baseRect;
}

const Texture2D& ButtonBase::GetTexture() const noexcept
{
	return this->baseTexture;
}

unsigned char Mouse::oldButtons;
unsigned char Mouse::buttons;
unsigned char Mouse::risingEdge;
unsigned char Mouse::fallingEdge;
glm::vec2 Mouse::position;