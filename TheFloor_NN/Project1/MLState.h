#pragma once

#include <vector>

struct NeighborState
{
	float myPower;
	float enemySpeed;
	float mySize;
	float enemySize;
};

struct MLState
{
	std::vector<NeighborState> neighbors;
};