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
	// TODO: Move to parallel once collision and more complicated stuff are done
	this->superDirty |= std::erase_if(this->debris, 
		[](Debris& ref) 
		{
			ref.transform.position += ref.delta.position * Tick::TimeDelta;
			ref.transform.rotation *= ref.delta.rotation;
			return glm::any(glm::isnan(ref.transform.position));
		}
	) != 0;
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
		//instanceVAO.ArrayFormatOverride<glm::mat4>("modelMat", shader, 1, 1);
		instanceVAO.ArrayFormatM<glm::mat4>(shader, 1, 1, "modelMat");
		CheckError();
		//instanceVAO.ArrayFormatOverride<glm::mat4>("normalMat", shader, 1, 1, sizeof(glm::mat4));
	}
	if (this->debris.size() == 0)
	{
		return;
	}
	shader.SetActiveShader();
	shader.SetVec3("shapeColor", glm::vec3(0.85, 0.25, 0.f));
	instanceVAO.Bind();
	instanceVAO.BindArrayBuffer(meshData.vertex, 0);
	instanceVAO.BindArrayBuffer(instanceBuffer, 1);
	meshData.index.BindBuffer();
	//shader.SetVec3("shapeColor", glm::vec3(0.85, 0.25, 0.f));
	//shader.DrawElementsInstanced<DrawType::Triangle>(meshData.index, instanceBuffer);
	//DrawIndirect thingy = meshData.rawIndirect[0];
	//thingy.instanceCount = this->debris.size();
	//shader.DrawElements<DrawType::Triangle>(thingy);
	meshData.index.BindBuffer();
	glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);
	meshData.indirect.BindBuffer();
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
		std::vector<std::pair<GLuint, GLuint>> offsets;
		offsets.reserve(meshData.rawIndirect.size());
		GLuint previous = 0, previousIndex = 0;
		for (GLuint i = 0; i < this->debris.size(); i++)
		{
			GLuint current = static_cast<GLuint>(this->debris[i].drawIndex);
			if (current != previous)
			{
				offsets.emplace_back(i - previousIndex, previousIndex);
				previousIndex = i;
				previous = current;
			}
		}
		while (offsets.size() < meshData.rawIndirect.size()) offsets.emplace_back(0, 0);
		for (std::size_t i = 0; i < meshData.rawIndirect.size(); i++)
		{
			auto& current = meshData.rawIndirect[i];
			current.instanceCount = offsets[i].first;
			current.instanceOffset = offsets[i].second;
		}
		meshData.indirect.BufferSubData(meshData.rawIndirect);
		this->superDirty = false;
		this->dirty = true;
	}
	if (this->dirty)
	{
		// Recalculate buffer
		std::vector<glm::mat4> matrix;
		matrix.reserve(this->debris.size());
		for (const auto& local : this->debris)
		{
			matrix.push_back(Model(local.transform).GetMatrixPair().model);
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
		rand() % debrisTypes
	);
	this->dirty = true;
	this->superDirty = true;
}

bool DebrisManager::LoadResources() noexcept
{
	meshData = OBJReader::ReadOBJSimple("Models\\Debris.obj");
	debrisTypes = meshData.indirect.GetElementCount() + 1;
	instanceBuffer.Generate();
	instanceBuffer.BufferData(glm::mat4(1.f));
	return debrisTypes > 1;
}
