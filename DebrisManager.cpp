#include "DebrisManager.h"
#include <glm/gtc/random.hpp>
#include "Buffer.h"
#include "OBJReader.h"
#include <numbers>
#include "VertexArray.h"
#include "Model.h"

static MeshData meshData;
static unsigned char debrisTypes = 1;
static VAO instanceVAO;

// Constant speed till 3 seconds after the inciting incident, then a decay for 3 seconds to whatever the final ones are
static constexpr std::uint16_t FadeOutTime     = static_cast<std::uint16_t>(Tick::PerSecond * 2.5);
static constexpr std::uint16_t FadeOutDuration = 2;
static constexpr std::uint16_t FinalSpeedTime  = FadeOutTime + static_cast<std::uint16_t>(Tick::PerSecond) * FadeOutDuration;
static constexpr float DecayConstant = 0.998f;

// A Full rotation every 5 seconds is the slowest I want it, for some reason
static constexpr float MinAngleRotation = Tick::TimeDelta * std::numbers::pi_v<float> * 2.f / 5.f;
static constexpr float MinSpeed = 0.25f;// *Tick::TimeDelta;
static constexpr float MaxSpeed = 3.5f;// *Tick::TimeDelta;

// TODO: Migrate to vector of vectors of Debris to remove the need for the stupid sorting and recalculations
// Might be able to get away with OctTree's instead of vectors but that kind of sounds like a headache of layers

void DebrisManager::Update() noexcept
{
	// TODO: Move to parallel once collision and more complicated stuff are done in here
	// TODO: Maybe this is good but I sure as hell can't tell, poke around numbers y'know
	std::size_t removedCount = std::erase_if(this->debris, 
		[](Debris& ref)
		{
			ref.ticksAlive++;
			ref.transform.position += ref.delta.position * Tick::TimeDelta;
			ref.transform.rotation = ref.transform.rotation * ref.delta.rotation;
			if (ref.ticksAlive == FinalSpeedTime)
			{
				float speed = glm::length(ref.delta.position);
				if (speed > MaxSpeed)
				{
					ref.delta.position = glm::normalize(ref.delta.position) * MaxSpeed;
				}
			}
			if (ref.ticksAlive < FinalSpeedTime && ref.ticksAlive > FadeOutTime)
			{
				// This is overcomplicated
				float speed = glm::length(ref.delta.position);
				if (speed > MinSpeed)
				{
					ref.delta.position *= DecayConstant;
				}
				float angle = glm::angle(ref.delta.rotation);
				if (angle > MinAngleRotation)
				{
					glm::vec3 axis = glm::axis(ref.delta.rotation);
					ref.delta.rotation = glm::angleAxis(angle * DecayConstant, axis);
					ref.delta.rotation = glm::normalize(ref.delta.rotation);
					if (glm::angle(ref.delta.rotation) > angle)
					{
						ref.delta.rotation = glm::angleAxis(MinAngleRotation, axis);
					}
				}
			}
			return glm::any(glm::isnan(ref.transform.position));
		}
	);
	this->superDirty |= (removedCount != 0);
	this->dirty = true;
}

void DebrisManager::Draw(Shader& shader) noexcept
{
	// TODO: Unhack this
	if (instanceVAO.GetArray() == 0)
	{
		//instanceVAO.ArrayFormat<MeshVertex>(shader);
		instanceVAO.ArrayFormatOverride<glm::vec3>(0, 0, 0, 0);
		instanceVAO.ArrayFormatOverride<glm::vec3>(1, 0, 0, offsetof(MeshVertex, normal));
		instanceVAO.ArrayFormatOverride<glm::vec2>(2, 0, 0, offsetof(MeshVertex, texture));
		instanceVAO.ArrayFormatOverride<glm::mat4>("modelMat", shader, 1, 1, 0, sizeof(MeshMatrix));
		instanceVAO.ArrayFormatOverride<glm::mat4>("normalMat", shader, 1, 1, sizeof(glm::mat4), sizeof(MeshMatrix));
	}
	if (this->indirectBuffer.GetElementCount() == 0)
	{
		this->indirectBuffer.BufferData(meshData.rawIndirect, StreamDraw);
	}
	if (this->debris.size() == 0)
	{
		return;
	}
	shader.SetActiveShader();
	shader.SetVec3("shapeColor", glm::vec3(0.85, 0.25, 0.f));
	shader.SetVec3("shapeColor", glm::vec3(0.85f));
	instanceVAO.Bind();
	instanceVAO.BindArrayBuffer(meshData.vertex, 0);
	instanceVAO.BindArrayBuffer(this->instanceBuffer, 1);
	meshData.index.BindBuffer();
	shader.MultiDrawElements(this->indirectBuffer);
}

void DebrisManager::FillBuffer() noexcept
{
	// Recalculate whole offset dealio
	if (this->superDirty)
	{
		std::sort(this->debris.begin(), this->debris.end(), 
			[] (const Debris& left, const Debris& right)
			{
				return left.drawIndex < right.drawIndex;
			});
		std::vector localCopy{ meshData.rawIndirect };
		// TODO: maybe do this on gpu or something I have no idea man
		std::vector<std::pair<GLuint, GLuint>> offsets(meshData.rawIndirect.size(), std::make_pair(0, 0));
		if (this->debris.size() > 0)
		{
			GLuint previous = this->debris[0].drawIndex, previousIndex = 0;
			for (GLuint i = 0; i < this->debris.size(); i++)
			{
				GLuint current = static_cast<GLuint>(this->debris[i].drawIndex);
				if (current != previous)
				{
					offsets[previous] = std::make_pair(i - previousIndex, previousIndex);
					previousIndex = i;
					previous = current;
				}
				if (static_cast<std::size_t>(i) + 1 == this->debris.size())
				{
					offsets[current] = std::make_pair(1 + i - previousIndex, previousIndex);
				}
			}
			for (std::size_t i = 0; i < meshData.rawIndirect.size(); i++)
			{
				auto& current = localCopy[i];
				current.instanceCount = offsets[i].first;
				current.instanceOffset = offsets[i].second;
			}
		}
		if (!this->indirectBuffer.GetBuffer())
		{
			this->indirectBuffer.BufferData(localCopy, StreamDraw);
		}
		else
		{
			this->indirectBuffer.BufferSubData(localCopy);
		}
		this->superDirty = false;
		this->dirty = true;
	}
	if (this->dirty)
	{
		// Recalculate buffer
		std::vector<MeshMatrix> matrix;
		matrix.reserve(this->debris.size());
		for (const auto& local : this->debris)
		{
			matrix.push_back(Model(local.transform, local.scale).GetMatrixPair());
		}
		this->instanceBuffer.BufferData(matrix, StreamDraw);
		this->dirty = false;
	}
}

void DebrisManager::AddDebris(glm::vec3 postion, glm::vec3 velocity) noexcept
{
	// TODO: Insertion sort based on id to add variety, yknow
	this->debris.emplace_back(Transform
		{
			postion,
			glm::quat(glm::sphericalRand(glm::pi<float>()))
		},
		Transform{
			velocity,
			glm::angleAxis(glm::radians(glm::length(velocity)), glm::sphericalRand(1.f))
		},
		static_cast<unsigned char>(rand()) % debrisTypes,
		static_cast<std::uint16_t>(this->debris.size() % 40), // Add some variation to the time out
		glm::clamp(glm::gaussRand(0.5f, 0.2f), 0.1f, 0.9f)
	);
	this->dirty = true;
	this->superDirty = true;
}

void DebrisManager::Add(std::vector<Debris>&& local) noexcept
{
	if (local.size() > 0)
	{
		this->dirty = true;
		this->superDirty = true;
	}
	std::move(local.begin(), local.end(), std::back_inserter(this->debris));
	local.clear();
}

void DebrisManager::Add(std::vector<Debris>& local) noexcept
{
	if (local.size() > 0)
	{
		this->dirty = true;
		this->superDirty = true;
	}
	std::move(local.begin(), local.end(), std::back_inserter(this->debris));
	local.clear();
}

bool DebrisManager::LoadResources() noexcept
{
	//meshData = OBJReader::ReadOBJSimple("Models\\Debris.obj");
	meshData = OBJReader::MeshThingy("Models\\Debris.obj");
	debrisTypes = meshData.indirect.GetElementCount();
	return debrisTypes > 1;
}
