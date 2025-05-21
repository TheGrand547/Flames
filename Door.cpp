#include "Door.h"
#include "Shader.h"
#include "ResourceBank.h"
#include "VertexArray.h"
#include "imgui/imgui.h"

static constexpr std::string_view ResourceName = "door_shader";

Door::Door(glm::vec3 position, State state, Status status) noexcept  : position(position), openState(state), lockedState(status) 
{

}

void Door::Update() noexcept
{
	if (this->openState == Door::Opening)
	{
		this->openTicks--;
		if (!this->openTicks)
		{
			this->openState = Door::Open;
		}
	}
	else if (this->openState == Door::Closing)
	{
		this->openTicks++;
		if (this->openTicks >= this->closingDuration)
		{
			this->openState = Door::Closed;
		}
	}
	// Hacked in for it too loop
	if (this->openState == Door::Open)
	{
		if (this->openTicks < -this->closingDuration)
		{
			this->openTicks = 0;
			this->openState = Door::Closing;
		}
		else
		{
			this->openTicks--;
		}
	}
	if (this->openState == Door::Closed)
	{
		if (this->openTicks > 2 * this->closingDuration)
		{
			this->openTicks = this->closingDuration;
			this->openState = Door::Opening;
		}
		else
		{
			this->openTicks++;
		}
	}
}

void Door::Draw() noexcept
{
	if (this->openState == Door::Closed)
	{
		// Trivial, draw the door as normal
		//ArrayBuffer& data = Bank<ArrayBuffer>::Get("door_shader");
	}
	else if (this->openState == Door::Open)
	{
		// Trivial, don't draw the door
		return;
	}
	else // Opening/closing
	{
		float progress = static_cast<float>(this->openTicks) / this->closingDuration;
		// Apply easing to progress
		// Could have different styles of doors, but thinking of the 'trivial' triangle one mainly
		// Wait the basic quad one is easier
		// Don't have to do all this logic, just get the progress and multiply vertices 2/3 x or y by it
	}


	ArrayBuffer& data = Bank<ArrayBuffer>::Get(ResourceName);
	struct local
	{
		glm::mat4 model{ 1.f }, normal{ 1.f };
		glm::vec2 layout{};
	} gimble;
	// I hate how screwed up this crap is, fix it and standardize it ffs
	gimble.model = glm::scale(glm::mat4(1.f), glm::vec3(4.f));
	if (this->openState == Door::Closed)
	{
		gimble.layout = glm::vec2(1.f);
	}
	else
	{
		float progress = std::clamp(static_cast<float>(this->openTicks) / this->closingDuration, 0.f, 1.f);
		// Logic depending upon major axis

		// Progress == 1, => door is closed
		// Progress == 0, => door is open
		gimble.layout = glm::vec2(progress, 1.f);
	}
	/*
	static float doorOpen = 0.5f;
	ImGui::Begin("Door");
	ImGui::SliderFloat("Door", &doorOpen, 0.f, 1.f);
	ImGui::End();
	gimble.layout.x = 1.f - doorOpen;
	*/

	if (data.GetBuffer() == 0)
	{
		data.BufferData(gimble);
	}

	data.BufferSubData(gimble);
	Shader& shader = ShaderBank::Get(ResourceName);
	VAO& vao = VAOBank::Get(ResourceName);
	shader.SetActiveShader();
	vao.Bind();
	vao.BindArrayBuffer(data);
	shader.SetVec3("shapeColor", glm::vec3(1.f, 1.f, 0.f));
	shader.SetTextureUnit("textureColor", Bank<Texture2D>::Get(ResourceName), 0);
	shader.DrawArray(6);
}

void Door::Setup()
{
	{
		Shader& ref = ShaderBank::Get(ResourceName);
		ref.Compile("door_simple", "deferred");
		ref.UniformBlockBinding("Camera", 0);
	}
	{
		Shader& shaderRef = ShaderBank::Get(ResourceName);
		VAO& ref = VAOBank::Get(ResourceName);
		ref.ArrayFormatOverride<glm::mat4>("modelMat", shaderRef, 0, 1, 0, 136ull);
		ref.ArrayFormatOverride<glm::mat4>("normalMat", shaderRef, 0, 1, sizeof(glm::mat4), 136ull);
		ref.ArrayFormatOverride<glm::vec2>("multiplier", shaderRef, 0, 1, 2 * sizeof(glm::mat4), 136ull);
	}
	{
		Texture2D& ref = Bank<Texture2D>::Get(ResourceName);
		ref.Load("text.png");
		ref.SetFilters();
	}
}
