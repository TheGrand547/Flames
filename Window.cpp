#include "Window.h"

namespace Window
{
    glm::mat4 GetMatInternal(float zNear, float zFar)
    {
        glm::mat4 grep = glm::perspective(GetYFOV(), AspectRatio, zFar, zNear);
        grep[2][2] = 0.f;
        grep[2][3] = -1.f;
        grep[3][2] = zNear;
        return grep;
    }
};