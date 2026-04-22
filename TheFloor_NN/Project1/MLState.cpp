#include "MLState.h"

#include <fstream>

void MLLogger::Log(MLState& state, int action, float reward, bool done)
{
	static std::ofstream file("ml/replay_buffer.csv", std::ios::app);

    std::array<float, FLAT_STATE_SIZE> flattenedState = state.Flatten();

	for (float value : flattenedState)
	{
        file << value << ",";
	}

	file << action << "," << reward << "," << done << "\n";
}

std::array<float, FLAT_STATE_SIZE> MLState::Flatten()
{
    std::array<float, FLAT_STATE_SIZE> flat{};
    flat.fill(0.0f);  // padding

    int index = 0;

    for (int i = 0; i < neighbors.size() && i < MAX_NEIGHBORS; i++)
    {
        flat[index++] = neighbors[i].myPower;
        flat[index++] = neighbors[i].enemySpeed;
        flat[index++] = neighbors[i].mySize;
        flat[index++] = neighbors[i].enemySize;
    }

    return flat;
}
