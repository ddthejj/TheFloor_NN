#include "MLState.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace
{
    std::filesystem::path GetReplayPath()
    {
        const char* configuredPath = std::getenv("THE_FLOOR_REPLAY_PATH");

        if (configuredPath != nullptr && configuredPath[0] != '\0')
        {
            return std::filesystem::path(configuredPath);
        }

        return std::filesystem::path("ml") / "replay_buffer.csv";
    }

    std::ofstream& GetReplayStream()
    {
        static std::ofstream file;
        static bool initialized = false;

        if (!initialized)
        {
            initialized = true;

            std::filesystem::path replayPath = GetReplayPath();
            std::error_code err;

            if (replayPath.has_parent_path())
            {
                std::filesystem::create_directories(replayPath.parent_path(), err);
                if (err)
                {
                    std::cerr << "[MLLogger] Failed to create directory '"
                              << replayPath.parent_path().string() << "': "
                              << err.message() << "\n";
                }
            }

            file.open(replayPath, std::ios::app);

            if (!file.is_open())
            {
                std::cerr << "[MLLogger] Failed to open replay log file: "
                          << replayPath.string() << "\n";
            }
            else
            {
                std::cout << "[MLLogger] Writing replay data to: "
                          << std::filesystem::absolute(replayPath).string() << "\n";
            }
        }

        return file;
    }
}

void MLLogger::Log(MLState& state, int action, float reward, bool done)
{
    std::ofstream& file = GetReplayStream();

    if (!file.is_open())
    {
        return;
    }

    std::array<float, FLAT_STATE_SIZE> flattenedState = state.Flatten();

    for (float value : flattenedState)
    {
        file << value << ",";
    }

    file << action << "," << reward << "," << done << "\n";
    file.flush();
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
