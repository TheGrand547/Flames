#pragma once
#ifndef SATELITE_H
#define SATELITE_H
#include "glmHelp.h"
#include "Transform.h"
#include "OBJReader.h"
#include "Shader.h"
#include "Capsule.h"

class Satelite
{
protected:
	Transform transform;
	float solarAngle;

public:
	Satelite(const Transform& transform) noexcept : transform(transform), solarAngle(0.f) {}

	void Draw(Shader& shader) const noexcept;
	void Update() noexcept;
	
	Capsule GetBounding() const noexcept;

	static bool LoadResources() noexcept;
};
#endif // SATELITE_H
