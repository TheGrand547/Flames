#include "Input.h"

namespace Input
{
	unsigned char Mouse::oldButtons;
	unsigned char Mouse::buttons;
	unsigned char Mouse::risingEdge;
	unsigned char Mouse::fallingEdge;
	glm::vec2 Mouse::position;
};