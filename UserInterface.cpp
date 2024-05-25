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

void Context::AddButton(std::shared_ptr<ButtonBase> button)
{
	this->elements.push_back(button);
}

void Context::Clear() noexcept
{
	this->elements.clear();
}

void Context::Update()
{
	for (auto& button : this->elements)
	{
		button->MouseUpdate();
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
