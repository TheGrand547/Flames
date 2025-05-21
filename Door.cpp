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
	// Could also return/not return the struct for instancing
	if (this->openState == Door::Open)
	{
		// Trivial, don't draw the door
		return;
	}
	ArrayBuffer& data = Bank<ArrayBuffer>::Get(ResourceName);
	struct local
	{
		glm::mat4 model{ 1.f }, normal{ 1.f };
		glm::vec2 layout{};
	} gimble;
	// Layout is {Progress, Type}
	// Progress == 1, => door is closed
	// Progress == 0, => door is open
	// Type == 1.f    => door is a triangle door
	// Type != 1.f    => door is a square door
	//gimble.model = glm::scale(glm::mat4(1.f), glm::vec3(4.f));
	gimble.model = this->model.GetModelMatrix();
	gimble.normal = this->model.GetNormalMatrix();
	if (this->openState == Door::Closed)
	{
		gimble.layout.x = 1.;
	}
	else
	{
		float progress = std::clamp(static_cast<float>(this->openTicks) / this->closingDuration, 0.f, 1.f);
		// Logic depending upon major axis

		// Progress == 1, => door is closed
		// Progress == 0, => door is open
		gimble.layout.x = progress;
	}
	gimble.layout.y = (this->openStyle == Type::Triangle) ? 1.f : 0.f;

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

// Colossal mess
std::array<Triangle, 2> Door::GetTris() const noexcept
{
	if (this->openState == Door::Open)
	{
		return std::to_array({Triangle(glm::mat3(-1000000.f)), Triangle(glm::mat3(-1000000.f))});
	}
	else 
	{
		// We're pretending that +x in the model axis corresponds to +y in 'real' space, will apply the transformation elsewhere
		const glm::mat3 axes = glm::mat3_cast(this->model.rotation);
		const glm::vec3 center = this->model.translation;
		const glm::vec3 scale = this->model.scale;

		const glm::vec3 axisA = axes[1];
		const glm::vec3 axisB = axes[2];

		float progress = std::clamp(static_cast<float>(this->openTicks) / this->closingDuration, 0.f, 1.f);
		glm::vec3 C = center + axisA * scale[1] + axisB * scale[2];
		glm::vec3 D = center + axisA * scale[1] - axisB * scale[2];

		glm::vec3 B = center - axisA * scale[1] + axisB * scale[2];
		glm::vec3 A = center - axisA * scale[1] - axisB * scale[2];
		if (this->openState == Door::Closed)
		{
			return std::to_array({ Triangle(B, A, C), Triangle(C,A, D) });
		}
		if (this->openStyle == Type::Square)
		{
			progress = 1.f + -2.f * progress;
			glm::vec3 CD = center - progress * axisA * scale[1] + axisB * scale[2];
			glm::vec3 DC = center - progress * axisA * scale[1] - axisB * scale[2];
			return std::to_array({ Triangle(B, A, CD), Triangle(CD, A, DC) });
		}
		else
		{
			progress = 1.f - progress;
			glm::vec3 C2 = center + progress * axisA * scale[1] + axisB * scale[2];
			glm::vec3 A2 = center - axisA * scale[1] - progress * axisB * scale[2];

			progress = 1.f - progress;
			glm::vec3 C3 = center + axisA * scale[1] + progress * axisB * scale[2];
			glm::vec3 A3 = center - progress * axisA * scale[1] - axisB * scale[2];
			return std::to_array({ Triangle(B, A2, C2), Triangle(C3,A3, D) });
		}
	}
}


