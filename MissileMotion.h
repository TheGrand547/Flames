#pragma once
#ifndef MISSILE_MOTION_H
#define MISSILE_MOTION_H
#include "glmHelp.h"

glm::vec3 MakePrediction(glm::vec3 thisPosition, glm::vec3 thisVelocity, float acceleration,
	glm::vec3 otherPosition, glm::vec3 otherVelocity);

#endif // MISSILE_MOTION_H