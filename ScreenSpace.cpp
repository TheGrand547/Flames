#include "ScreenSpace.h"

static Shader screenShader;
static UniformBuffer screenProjection;

Shader& ScreenSpace::GetShader()
{
    return screenShader;
}

UniformBuffer& ScreenSpace::GetProjection()
{
    return screenProjection;
}

void Setup()
{
    
}
