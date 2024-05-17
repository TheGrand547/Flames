#include "Button.h"

ScreenRect ButtonBase::GetRect() const noexcept
{
	return this->baseRect;
}

const Texture2D& ButtonBase::GetTexture() const noexcept
{
	return this->baseTexture;
}