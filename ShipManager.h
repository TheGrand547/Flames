#pragma once
#ifndef SHIP_MANAGER_H
#define SHIP_MANAGER_H
#include "DynamicTree.h"
#include "ClockBrain.h"

class ShipManager
{
protected:
	DynamicOctTree<ClockBrain> brainDrain;
};

#endif // SHIP_MANAGER_H