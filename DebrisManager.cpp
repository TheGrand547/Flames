#include "DebrisManager.h"
#include <glm/gtc/random.hpp>
#include "Buffer.h"
#include "OBJReader.h"
#include <numbers>
#include "VertexArray.h"
#include "Model.h"
#include "ResourceBank.h"
#include "Input.h"
#include <ranges>
#include <execution>
#include "StaticVector.h"
#include "Parallel.h"

static MeshData meshData;
static unsigned char DebrisTypes = 1;

// Constant speed till 3 seconds after the inciting incident, then a decay for 3 seconds to whatever the final ones are
static constexpr std::uint16_t FadeOutTime     = static_cast<std::uint16_t>(Tick::PerSecond * 2.5);
static constexpr std::uint16_t FadeOutDuration = 2;
static constexpr std::uint16_t FinalSpeedTime  = FadeOutTime + static_cast<std::uint16_t>(Tick::PerSecond) * FadeOutDuration;
static constexpr std::uint16_t DecayStart      = FinalSpeedTime + static_cast<std::uint16_t>(Tick::PerSecond) * 2;
static constexpr std::uint16_t DecayLength     = static_cast<std::uint16_t>(Tick::PerSecond);
static constexpr std::uint16_t DecayEnd        = DecayStart + DecayLength;
static constexpr float DecayConstant = 0.998f;

// A Full rotation every 5 seconds is the slowest I want it, for some reason
static constexpr float MinAngleRotation = Tick::TimeDelta * std::numbers::pi_v<float> * 2.f / 5.f;
static constexpr float MinSpeed = 0.25f;// *Tick::TimeDelta;
static constexpr float MaxSpeed = 3.5f;// *Tick::TimeDelta;

void DebrisManager::Update() noexcept
{
	// This parallel one is faster for larger groups of them(shocking), will have to do more testing to determine if it's worth it
	// This is vaguely in the ballpark of successful
	if (Input::Mouse::CheckButton(Input::Mouse::ButtonMiddle) || this->elementCount > 100) 
	{
		//std::vector<std::vector<MeshMatrix>> evil;
		/*
		for (auto i = 0; i < DebrisTypes; i++)
		{
			evil.push_back({});
		}*/
		StaticVector<std::vector<MeshMatrix>> evil(DebrisTypes);
		//std::ranges::iota_view viewing(static_cast<std::size_t>(0), static_cast<std::size_t>(DebrisTypes));
		std::atomic<std::size_t> removedCount = 0;
		//std::for_each(std::execution::par, viewing.begin(), viewing.end(), [&](size_t i)
		Parallel::for_each_index(std::execution::par, evil, [&](std::size_t i )
			{
				removedCount += std::erase_if(this->debris[i],
					[&](Debris& ref)
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
						if (ref.ticksAlive > DecayStart)
						{
							if (ref.ticksAlive > DecayEnd)
							{
								//return true;
							}
							ref.scale *= 0.95f;
						}
						else if (ref.ticksAlive < FinalSpeedTime && ref.ticksAlive > FadeOutTime)
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
						bool result = glm::any(glm::isnan(ref.transform.position));
						if (!result)
						{
							evil[i].push_back(Model(ref.transform, ref.scale).GetMatrixPair());
						}
						return result;
					}
				);
			}
			);
		this->elementCount -= removedCount;
		std::vector<MeshMatrix> cluster;
		cluster.reserve(this->elementCount);
		for (auto& ref : evil)
		{
			std::move(ref.begin(), ref.end(), std::back_inserter(cluster));
		}
		this->buffered.Swap(cluster);
		this->superDirty |= (removedCount != 0);
		this->dirty = true;
	}
	else
	{
		std::vector<MeshMatrix> paringKnife;
		paringKnife.reserve(this->elementCount);
		// TODO: Move to parallel once collision and more complicated stuff are done in here
		// TODO: Maybe this is good but I sure as hell can't tell, poke around numbers y'know


		std::size_t removedCount = 0;

		for (auto& subArray : this->debris)
		{
			removedCount += std::erase_if(subArray,
				[&](Debris& ref)
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
					if (ref.ticksAlive > DecayStart)
					{
						if (ref.ticksAlive > DecayEnd)
						{
							//return true;
						}
						ref.scale *= 0.95f;
					}
					else if (ref.ticksAlive < FinalSpeedTime && ref.ticksAlive > FadeOutTime)
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
					bool result = glm::any(glm::isnan(ref.transform.position));
					if (!result)
					{
						paringKnife.push_back(Model(ref.transform, ref.scale).GetMatrixPair());
					}
					return result;
				}
			);
		}
		this->elementCount -= removedCount;
		this->buffered.Swap(paringKnife);
		this->superDirty |= (removedCount != 0);
		this->dirty = true;
	}
}

void DebrisManager::Draw(Shader& shader) noexcept
{
	// TODO: Unhack this
	VAO& instanceVAO = VAOBank::Get("instance");
	if (instanceVAO.GetArray() == 0)
	{
		//instanceVAO.ArrayFormat<MeshVertex>(shader);
		instanceVAO.ArrayFormatOverride<glm::vec3>(0, 0, 0, 0);
		instanceVAO.ArrayFormatOverride<glm::vec3>(1, 0, 0, offsetof(MeshVertex, normal));
		instanceVAO.ArrayFormatOverride<glm::vec2>(2, 0, 0, offsetof(MeshVertex, texture));
		instanceVAO.ArrayFormatOverride<glm::mat4>("modelMat", shader, 1, 1, 0, sizeof(MeshMatrix));
		instanceVAO.ArrayFormatOverride<glm::mat4>("normalMat", shader, 1, 1, sizeof(glm::mat4), sizeof(MeshMatrix));
		
		/*
		instanceVAO.ArrayFormatOverride<glm::vec3>(0, 0, 0, 0);
		instanceVAO.ArrayFormatOverride<glm::vec3>(1, 0, 0, offsetof(NormalMeshVertex, normal));
		instanceVAO.ArrayFormatOverride<glm::vec3>(2, 0, 0, offsetof(NormalMeshVertex, tangent));
		instanceVAO.ArrayFormatOverride<glm::vec3>(3, 0, 0, offsetof(NormalMeshVertex, biTangent));
		instanceVAO.ArrayFormatOverride<glm::vec2>(4, 0, 0, offsetof(NormalMeshVertex, texture));
		instanceVAO.ArrayFormatOverride<glm::mat4>("modelMat", shader, 1, 1, 0, sizeof(MeshMatrix));
		instanceVAO.ArrayFormatOverride<glm::mat4>("normalMat", shader, 1, 1, sizeof(glm::mat4), sizeof(MeshMatrix));
		*/
	}
	if (this->indirectBuffer.GetElementCount() == 0)
	{
		this->indirectBuffer.BufferData(meshData.rawIndirect, StreamDraw);
	}
	if (this->elementCount == 0)
	{
		return;
	}
	shader.SetActiveShader();
	//shader.SetVec3("shapeColor", glm::vec3(0.85, 0.25, 0.f));
	shader.SetVec3("shapeColor", glm::vec3(0.85f));
	instanceVAO.Bind();
	instanceVAO.BindArrayBuffer(meshData.vertex, 0);
	instanceVAO.BindArrayBuffer(this->instanceBuffer, 1);
	meshData.index.BindBuffer();
	shader.MultiDrawElements(this->indirectBuffer);
}

void DebrisManager::Init() noexcept
{
	this->debris.clear();
	for (auto i = 0; i < DebrisTypes; i++)
	{
		this->debris.push_back({});
	}
}

void DebrisManager::FillBuffer() noexcept
{
	// Recalculate whole offset dealio
	if (this->superDirty)
	{
		std::vector<DrawIndirect> localCopy = meshData.rawIndirect;
		if (this->debris.size() > 0)
		{
			GLuint offset = 0;
			for (auto i = 0; i < this->debris.size(); i++)
			{
				localCopy[i].instanceCount = static_cast<GLuint>(this->debris[i].size());
				localCopy[i].instanceOffset = offset;
				offset += static_cast<GLuint>(this->debris[i].size());
			}
			this->elementCount = offset;
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
		this->buffered.ExclusiveOperation([&](std::vector<MeshMatrix>& matrix)
			{
				this->instanceBuffer.BufferData(matrix, StreamDraw);
			}
		);
		this->dirty = false;
	}
}

void DebrisManager::AddDebris(glm::vec3 postion, glm::vec3 velocity) noexcept
{
	unsigned char index = static_cast<unsigned char>(rand()) % DebrisTypes;
	this->elementCount += 1;
	this->debris[index].emplace_back(Transform
		{
			postion,
			glm::quat(glm::sphericalRand(glm::pi<float>()))
		},
		Transform{
			velocity,
			glm::angleAxis(glm::radians(glm::length(velocity)), glm::sphericalRand(1.f))
		},
		index,
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
	this->elementCount += local.size();
	for (auto& debris : local)
	{
		this->debris[debris.drawIndex].push_back(std::move(debris));
	}
	local.clear();
}

void DebrisManager::Add(std::vector<Debris>& local) noexcept
{
	if (local.size() > 0)
	{
		this->dirty = true;
		this->superDirty = true;
	}
	this->elementCount += local.size();
	for (const auto& debris: local)
	{
		this->debris[debris.drawIndex].push_back(debris);
	}
	local.clear();
}

bool DebrisManager::LoadResources() noexcept
{
	//meshData = OBJReader::MeshThingy<NormalMeshVertex>("Models\\Debris.glb");
	meshData = OBJReader::MeshThingy<MeshVertex>("Models\\Debris.glb");
	DebrisTypes = meshData.indirect.GetElementCount();
	return DebrisTypes > 1;
}
