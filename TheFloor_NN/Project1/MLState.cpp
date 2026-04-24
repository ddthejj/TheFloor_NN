#include "MLState.h"

#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

namespace
{
    std::string GetReplayPath()
    {
        const char* envPath = std::getenv("THE_FLOOR_REPLAY_PATH");
        if (envPath != nullptr && envPath[0] != '\0')
        {
            return envPath;
        }

        return "ml/replay_buffer.csv";
    }

    std::string GetParentDirectory(const std::string& filePath)
    {
        std::size_t slashPos = filePath.find_last_of("/\\");
        if (slashPos == std::string::npos)
        {
            return "";
        }

        return filePath.substr(0, slashPos);
    }

    bool CreateDirectorySingle(const std::string& directoryPath)
    {
#ifdef _WIN32
        int result = _mkdir(directoryPath.c_str());
#else
        int result = mkdir(directoryPath.c_str(), 0755);
#endif

        return result == 0 || errno == EEXIST;
    }

    bool DirectoryExistsOrCreate(const std::string& directoryPath)
    {
        if (directoryPath.empty())
        {
            return true;
        }

        std::string current;
        for (std::size_t i = 0; i < directoryPath.size(); ++i)
        {
            char ch = directoryPath[i];
            current.push_back(ch);

            if ((ch == '/' || ch == '\\') && !current.empty())
            {
                if (!CreateDirectorySingle(current))
                {
                    return false;
                }
            }
        }

        return CreateDirectorySingle(directoryPath);
    }

    std::ofstream& GetReplayStream()
    {
        static std::ofstream file;
        static bool initialized = false;

        if (!initialized)
        {
            initialized = true;

            std::string replayPath = GetReplayPath();
            std::string replayDir = GetParentDirectory(replayPath);

            if (!DirectoryExistsOrCreate(replayDir))
            {
                std::cerr << "[MLLogger] Failed to create directory '" << replayDir << "'\n";
            }

            file.open(replayPath.c_str(), std::ios::app);

            if (!file.is_open())
            {
                std::cerr << "[MLLogger] Failed to open replay log file: "
                          << replayPath << "\n";
            }
            else
            {
                bool hasExistingRows = false;
                std::ifstream existingFile(replayPath.c_str());
                if (existingFile.good())
                {
                    hasExistingRows = existingFile.peek() != std::ifstream::traits_type::eof();
                }

                if (!hasExistingRows)
                {
                    std::ostringstream header;

                    for (int neighborIndex = 0; neighborIndex < MAX_NEIGHBORS; ++neighborIndex)
                    {
                        header << "neighbor_" << neighborIndex << "_myPower,";
                        header << "neighbor_" << neighborIndex << "_enemySpeed,";
                        header << "neighbor_" << neighborIndex << "_mySize,";
                        header << "neighbor_" << neighborIndex << "_enemySize,";
                        header << "neighbor_" << neighborIndex << "_isChallengeable,";
                    }

                    header << "action,reward,done\n";
                    file << header.str();
                    file.flush();
                }

                std::cout << "[MLLogger] Writing replay data to: "
                          << replayPath << "\n";
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
        flat[index++] = neighbors[i].isChallengeable;
    }

    return flat;
}
