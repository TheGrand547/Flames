#include "UserInterface.h"
#include "Shader.h"
static Shader buttonShader;

static void SetupButtonShader()
{
	if (!buttonShader.Compiled())
	{
		buttonShader.CompileSimple("ui_rect_texture");
		buttonShader.UniformBlockBinding("ScreenSpace", 1);
	}
	buttonShader.SetActiveShader();
}

Context::~Context() noexcept
{
	this->Clear();
}

void Context::AddButton(ButtonBase* button)
{
	this->elements.push_back(button);
}

void Context::Clear() noexcept
{
	for (std::size_t i = 0; i < this->elements.size(); i++)
	{
		delete this->elements[i];
		this->elements[i] = nullptr;
	}
	this->elements.clear();
}

void Context::Update(MouseStatus& status)
{
	bool visualUpdate = false;
	for (auto& button : this->elements)
	{
		visualUpdate |= button->MouseUpdate(status);
	}
	if (visualUpdate)
	{
		// TODO: Visual Update
	}
}

void Context::Draw()
{
	SetupButtonShader();
	for (auto& button : this->elements)
	{
		// TODO: Options
		// TODO: Maybe instancing or something idk
		buttonShader.SetVec4("image", button->GetRect());
		buttonShader.SetTextureUnit("rectangle", button->GetTexture(), 0);
		buttonShader.DrawArray<DrawType::TriangleStrip>(4);
	}
}
