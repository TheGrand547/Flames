#include "Frustum.h"
#include "Window.h"



Frustum::Frustum(const glm::vec3& cameraPosition, const glm::quat& orientation, const glm::vec2& clippingPlanes) noexcept
{
    const glm::mat3 cameraBasis(orientation);
    const glm::vec2 halfEdges(clippingPlanes[1] * std::tanf(Window::GetYFOV()), 
        clippingPlanes[1] * std::tanf(Window::GetYFOV()) * Window::AspectRatio);
    const glm::vec3 farEdge = cameraBasis[0] * clippingPlanes[1];

    this->near = Plane(cameraBasis[0], cameraPosition + clippingPlanes[0] * cameraBasis[0]);
    this->far = Plane(-cameraBasis[0], farEdge);

    this->right = Plane(glm::cross(farEdge - cameraBasis[2] * halfEdges[1], cameraBasis[1]), cameraPosition);
    this->left = Plane(glm::cross(cameraBasis[1], farEdge + cameraBasis[2] * halfEdges[1]), cameraPosition);

    this->top = Plane(glm::cross(cameraBasis[2], farEdge + cameraBasis[1] * halfEdges[0]), cameraPosition);
    this->top = Plane(glm::cross(farEdge - cameraBasis[1] * halfEdges[0], cameraBasis[2]), cameraPosition);

}

bool Frustum::Overlaps(const Sphere& sphere) const noexcept
{
    return sphere.FrontOrCollide(this->near) && sphere.FrontOrCollide(this->far) 
        && sphere.FrontOrCollide(this->left) && sphere.FrontOrCollide(this->right)
        && sphere.FrontOrCollide(this->top) && sphere.FrontOrCollide(this->left);
}
