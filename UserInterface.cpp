#include "UserInterface.h"

Context::~Context()
{
	for (std::size_t i = 0; i < this->elements.size(); i++)
	{
		delete this->elements[i];
		this->elements[i] = nullptr;
	}
	this->elements.clear();
}

void Context::AddButton(ButtonBase* button)
{
	this->elements.push_back(button);
}

void Context::Update(MouseStatus& status)
{
	for (auto& button : this->elements)
	{
		button->MouseUpdate(status);
	}
}
