#pragma once
#ifndef DEMO_GUY_H
#define DEMO_GUY_H
#include "glmHelp.h"
#include "Capsule.h"



// Wowza encapsulation
class DemoGuy
{
private:
	glm::vec3 position;
	
public:


	static const glm::vec3 MidpointOffset;
};

#endif // DEMO_GUY_H