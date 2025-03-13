#include "ClockBrain.h"
#include <glm/gtc/random.hpp>
#include "Model.h"
#include "BasicPhysics.h"
#include "MissileMotion.h"
#include "log.h"

void ClockBrain::Init()
{
	this->transform.position = glm::ballRand(50.f);
	this->transform.rotation = glm::angleAxis(glm::gaussRand(glm::pi<float>(), glm::pi<float>()), glm::sphericalRand(1.f));
	this->velocity = this->transform.rotation * glm::vec3(1.f, 0, 0);
}

void ClockBrain::Update()
{
	if (glm::distance(this->transform.position, this->target) < 0.5)
	{
		// Generate new target
		this->target = glm::ballRand(100.f);
		Log("Switched targets");
	}
	glm::vec3 forced = MakePrediction(this->transform.position, this->velocity, 20.f, this->target, glm::vec3(0.f));
	BasicPhysics::Update(this->transform.position, this->velocity, forced);
	BasicPhysics::Clamp(this->velocity, 10.f);
	// Pretend it's non-zero
	if (glm::length(this->velocity) > EPSILON)
	{
		glm::mat3 orient{};
		glm::mat3 current = glm::mat3_cast(this->transform.rotation);
		orient[0] = glm::normalize(this->velocity);
		orient[1] = glm::cross(current[2], orient[0]);
		orient[2] = glm::cross(orient[0], orient[1]);
		this->transform.rotation = glm::quat_cast(orient);
	}
}

void ClockBrain::Draw(MeshData& data, VAO& vao, Shader& shader)
{
	shader.SetActiveShader();
	vao.Bind();
	vao.BindArrayBuffer(data.vertex);
	data.index.BindBuffer();
	Model model{ this->transform };
	shader.SetMat4("modelMat", model.GetModelMatrix());
	shader.SetMat4("normalMat", model.GetNormalMatrix());
	shader.DrawElements(data.indirect);
}
