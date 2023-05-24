#include "OrientedBoundingBox.h"

OrientedBoundingBox::~OrientedBoundingBox() {}

Model OrientedBoundingBox::GetModel() const
{
	glm::mat4 mat(glm::vec4(this->axes[0].first, 0), glm::vec4(this->axes[1].first, 0), 
					glm::vec4(this->axes[2].first, 0), glm::vec4(0, 0, 0, 1));
	glm::vec3 angles{ 0.f, 0.f, 0.f };
	glm::extractEulerAngleXYZ(mat, angles.x, angles.y, angles.z);
	return Model(this->center, glm::degrees(angles), glm::vec3(this->axes[0].second, this->axes[1].second, this->axes[2].second));
}
