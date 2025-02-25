#include "MissileMotion.h"
#include "util.h"

glm::vec3 MakePrediction(glm::vec3 thisPosition, glm::vec3 thisVelocity, float acceleration,
	glm::vec3 otherPosition, glm::vec3 otherVelocity)
{
	glm::vec3 direction = otherPosition - thisPosition;
	glm::vec3 relativeVelocity = otherVelocity - thisVelocity;

	// a
	float strength = acceleration;
	// b
	float percentage = glm::dot(-relativeVelocity, direction);
	// c
	float distance = Rectify(glm::length(direction));

	// Probably a more numerically stable way to do this but I don't care
	// Estimate coincidence along the delta axis, assuming constant acceleration along the acceleration axis
	// In the quadratic equation: 0.5*a*t^2 + b*t + c --- with: (a) = strength, (b) = percentage, (c) = -error
	
	// Since the distance is being solved for, in the physics equation , it gets a negative sign, thus making 
	// the -4ac into +4ac on the line below
	float descriminant = percentage * percentage + 4 * strength * distance;
	float inverse = 1.f / (2.f * strength);
	float result = (glm::sqrt(descriminant) - percentage) * inverse;
	
	glm::vec3 estimated = otherPosition + relativeVelocity * Rectify(result);
	glm::vec3 estimate = glm::normalize(estimated - thisPosition) * acceleration;
	if (glm::any(glm::isnan(estimate)))
	{
		std::cout << "God dammit\n";
	}
	return estimate;
}
