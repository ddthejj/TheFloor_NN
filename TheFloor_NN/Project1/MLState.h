#pragma once

#include <vector>
#include <array>

static constexpr int MAX_NEIGHBORS = 50;
static constexpr int FEATURES_PER_NEIGHBOR = 4;
static constexpr int FLAT_STATE_SIZE = MAX_NEIGHBORS * FEATURES_PER_NEIGHBOR;
static constexpr int STAYGO_STATE_SIZE = 8;

struct NeighborState
{
	float myPower;
	float enemySpeed;
	float mySize;
	float enemySize;
};

struct StayGoState
{
	std::array<float, STAYGO_STATE_SIZE> values;
};

struct MLState
{
	std::vector<NeighborState> neighbors;

	std::array<float, FLAT_STATE_SIZE> Flatten();
};

class MLLogger
{
public:
	static void Log(MLState& state, int action, float reward, bool done);
};