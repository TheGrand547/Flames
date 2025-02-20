#include "DebrisManager.h"
#include <glm/gtc/random.hpp>
#include "Buffer.h"
#include "OBJReader.h"
#include "VertexArray.h"
#include "Model.h"

static MeshData meshData;
static unsigned char debrisTypes = 1;
static VAO instanceVAO;
static ArrayBuffer instanceBuffer;

void DebrisManager::Update() noexcept
{
	// TODO: Move to parallel once collision and more complicated stuff are done in here
	// TODO: Proper decay constant
	std::size_t removedCount = std::erase_if(this->debris, 
		[](Debris& ref)
		{
			ref.transform.position += ref.delta.position * Tick::TimeDelta;
			ref.transform.rotation *= ref.delta.rotation;
			ref.delta.position *= 0.999f;
			float angle = glm::angle(ref.delta.rotation);
			ref.delta.rotation = glm::angleAxis(angle * 0.999f, glm::axis(ref.delta.rotation));
			ref.delta.rotation = glm::normalize(ref.delta.rotation);
			if (glm::angle(ref.delta.rotation) > angle)
			{
				ref.delta.rotation = QuatIdentity();
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
		CheckError();
		instanceVAO.ArrayFormat<MeshVertex>(shader);
		CheckError();
		instanceVAO.ArrayFormatOverride<glm::mat4>("modelMat", shader, 1, 1, 0, sizeof(MeshMatrix));
		//instanceVAO.ArrayFormatM<glm::mat4>(shader, 1, 1, "modelMat");
		instanceVAO.ArrayFormatOverride<glm::mat4>("normalMat", shader, 1, 1, sizeof(glm::mat4), sizeof(MeshMatrix));
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
	instanceVAO.BindArrayBuffer(instanceBuffer, 1);
	meshData.index.BindBuffer();
	shader.MultiDrawElements(meshData.indirect);
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
				auto& current = meshData.rawIndirect[i];
				current.instanceCount = offsets[i].first;
				current.instanceOffset = offsets[i].second;
			}
		}
		
		meshData.indirect.BufferSubData(meshData.rawIndirect);
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
			matrix.push_back(Model(local.transform).GetMatrixPair());
		}
		instanceBuffer.BufferData(matrix);
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
		static_cast<unsigned char>(rand()) % debrisTypes
		//this->debris.size() % debrisTypes
	);
	this->dirty = true;
	this->superDirty = true;
}

bool DebrisManager::LoadResources() noexcept
{
	meshData = OBJReader::ReadOBJSimple("Models\\Debris.obj");
	debrisTypes = meshData.indirect.GetElementCount();
	instanceBuffer.Generate();
	instanceBuffer.BufferData(glm::mat4(1.f));
	return debrisTypes > 1;
}
