#include "Tile.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>

#ifdef _WIN32
#define POPEN _popen
#define PCLOSE _pclose
#else
#define POPEN popen
#define PCLOSE pclose
#endif

namespace
{
	constexpr const char* kPredictScriptPath = "../NN_Training/predict_action.py";
	constexpr const char* kModelPath = "model/floor_ai.keras";
	constexpr const char* kNormPath = "model/floor_ai.norm.json";
	constexpr const char* kRequestPath = "model/predict_request.txt";
	constexpr const char* kResponsePath = "model/predict_response.txt";

	bool FileExists(const char* filePath)
	{
		std::ifstream file(filePath);
		return file.good();
	}

	std::string BuildFlatState(const std::array<float, FLAT_STATE_SIZE>& flatState)
	{
		std::ostringstream stateBuilder;
		for (int i = 0; i < FLAT_STATE_SIZE; ++i)
		{
			if (i > 0)
			{
				stateBuilder << ",";
			}

			stateBuilder << flatState[i];
		}

		return stateBuilder.str();
	}

	class PythonPredictServer
	{
	public:
		static PythonPredictServer& Get()
		{
			static PythonPredictServer server;
			return server;
		}

		int Predict(const std::array<float, FLAT_STATE_SIZE>& flatState, int validNeighborCount)
		{
			if (!EnsureStarted())
			{
				return -1;
			}

			++requestIdCounter;
			const std::uint64_t requestId = requestIdCounter;

			std::ofstream requestFile(kRequestPath, std::ios::trunc);
			if (!requestFile.is_open())
			{
				return -1;
			}

			requestFile
				<< requestId << "|"
				<< validNeighborCount << "|"
				<< BuildFlatState(flatState) << "\n";
			requestFile.close();

			const auto timeoutAt = std::chrono::steady_clock::now() + std::chrono::milliseconds(1500);
			while (std::chrono::steady_clock::now() < timeoutAt)
			{
				std::ifstream responseFile(kResponsePath);
				if (responseFile.is_open())
				{
					std::string responseLine;
					std::getline(responseFile, responseLine);
					responseFile.close();

					const std::size_t firstSep = responseLine.find('|');
					if (firstSep != std::string::npos)
					{
						const std::uint64_t responseId = static_cast<std::uint64_t>(std::strtoull(responseLine.substr(0, firstSep).c_str(), nullptr, 10));
						if (responseId == requestId)
						{
							return std::atoi(responseLine.substr(firstSep + 1).c_str());
						}
					}
				}

				std::this_thread::sleep_for(std::chrono::milliseconds(2));
			}

			return -1;
		}

	private:
		FILE* processHandle = nullptr;
		std::uint64_t requestIdCounter = 0;
		bool shutdownHookRegistered = false;

		static void ShutdownAtExit()
		{
			PythonPredictServer::Get().Shutdown();
		}

		bool EnsureStarted()
		{
			if (processHandle != nullptr)
			{
				return true;
			}

			std::ofstream(kRequestPath, std::ios::trunc).close();
			std::ofstream(kResponsePath, std::ios::trunc).close();

			std::ostringstream commandBuilder;
			commandBuilder
				<< "python \"" << kPredictScriptPath << "\""
				<< " --serve"
				<< " --model \"" << kModelPath << "\""
				<< " --request-file \"" << kRequestPath << "\""
				<< " --response-file \"" << kResponsePath << "\"";

			if (FileExists(kNormPath))
			{
				commandBuilder << " --norm \"" << kNormPath << "\"";
			}

#ifdef _WIN32
			commandBuilder << " >NUL 2>&1";
#else
			commandBuilder << " >/dev/null 2>&1";
#endif

			processHandle = POPEN(commandBuilder.str().c_str(), "r");
			if (processHandle == nullptr)
			{
				return false;
			}

			if (!shutdownHookRegistered)
			{
				std::atexit(&PythonPredictServer::ShutdownAtExit);
				shutdownHookRegistered = true;
			}

			const auto timeoutAt = std::chrono::steady_clock::now() + std::chrono::milliseconds(4000);
			while (std::chrono::steady_clock::now() < timeoutAt)
			{
				std::ifstream responseFile(kResponsePath);
				if (responseFile.is_open())
				{
					std::string line;
					std::getline(responseFile, line);
					responseFile.close();

					if (line == "READY")
					{
						return true;
					}
				}

				std::this_thread::sleep_for(std::chrono::milliseconds(5));
			}

			Shutdown();
			return false;
		}

		void Shutdown()
		{
			if (processHandle == nullptr)
			{
				return;
			}

			std::ofstream requestFile(kRequestPath, std::ios::trunc);
			if (requestFile.is_open())
			{
				requestFile << "QUIT\n";
			}

			PCLOSE(processHandle);
			processHandle = nullptr;
		}
	};

	int PredictActionFromModel(const std::array<float, FLAT_STATE_SIZE>& flatState, int validNeighborCount)
	{
		if (!FileExists(kModelPath))
		{
			return -1;
		}

		return PythonPredictServer::Get().Predict(flatState, validNeighborCount);
	}
}

Tile::Tile(int _category, int numCategories) : category(_category), player(_category, numCategories)
{
}

int Tile::GetPower(int battleCategory)
{
	// For simplicity, we're saying a player's "Power" is 75% knowledge and 25% speed.
	return (((float)player.GetSkill(battleCategory) * 7.5f) + ((float)player.GetSpeed() * (2.5f)));
}

void Tile::AddNeighbor(Tile* neighbor)
{
	if (neighbor != this)
	{
		if (std::find(neighbors.begin(), neighbors.end(), neighbor) == neighbors.end())
		{
			neighbors.push_back(neighbor);
			neighbor->AddNeighbor(this);
		}
	}
}

void Tile::RemoveNeighbor(Tile* neighbor)
{
	auto it = std::find(neighbors.begin(), neighbors.end(), neighbor);
	
	if (it != neighbors.end())
	{
		if (!neighbor->hasLost)
		{
			neighbor->RemoveNeighbor(this);
		}
		neighbors.erase(it);
	}
}

Tile* Tile::ChooseNeighbor()
{
	hasPlayed = true;

	MLState state = GetMLState();
	std::array<float, FLAT_STATE_SIZE> flatState = state.Flatten();

	int action = PredictActionFromModel(flatState, static_cast<int>(neighbors.size()));

	if (action < 0 || action >= static_cast<int>(neighbors.size()))
	{
		action = (std::rand() % neighbors.size());
	}

	StoreDecision(state, action);

	neighbors[action]->hasPlayed = true;

	return neighbors[action];
}

void Tile::WinBattle(bool wasAttacker, Tile* loserTile)
{
	if (wasAttacker)
	{

	}
	else
	{
		category = loserTile->GetCategory();
	}

	for (int i = 0; i < loserTile->neighbors.size(); i++)
	{
		AddNeighbor(loserTile->neighbors[i]);
	}

	size++;
}

void Tile::LoseBattle()
{
	hasLost = true;

	auto it = neighbors.begin();

	while (it != neighbors.end())
	{
		(*it)->RemoveNeighbor(this);
		it++;
	}

	neighbors.clear();
	size = 0;
}

MLState Tile::GetMLState()
{
	MLState state;

	for (Tile* neighbor : neighbors)
	{
		NeighborState neighborState;
		const int battleCategory = neighbor->GetCategory();
		const float myBattlePower = GetPower(battleCategory);

		neighborState.myPower = myBattlePower;
		neighborState.enemySpeed = neighbor->hasPlayed ? neighbor->GetPlayer()->GetSpeed() : -10;
		neighborState.mySize = size;
		neighborState.enemySize = neighbor->size;
		neighborState.isChallengeable = 0.0f;

		if (neighbor->hasPlayed)
		{
			const float enemyBattlePower = neighbor->GetPower(battleCategory);
			neighborState.isChallengeable = myBattlePower >= enemyBattlePower ? 1.0f : 0.0f;
		}

		state.neighbors.push_back(neighborState);
	}

	return state;
}

void Tile::StoreDecision(MLState& state, int action)
{
	lastState = state;
	lastAction = action;
	hasPendingDecision = true;
}

void Tile::ResolveDecision(float reward, bool done)
{
	if (!hasPendingDecision)
	{
		return;
	}

	MLLogger::Log(lastState, lastAction, reward, done);
	hasPendingDecision = false;
}

StayGoState Tile::GetStayGoState()
{
	StayGoState state;

	state.values.fill(0.0f);

	MLState mlState = GetMLState();
	const std::array<float, FLAT_STATE_SIZE> flatState = mlState.Flatten();

	float challengeableCount = 0.0f;
	float knownEnemySpeedCount = 0.0f;

	for (int i = 0; i < MAX_NEIGHBORS; ++i)
	{
		const int baseIndex = i * FEATURES_PER_NEIGHBOR;
		if (flatState[baseIndex] <= 0.0f)
		{
			continue;
		}

		if (flatState[baseIndex + 4] > 0.5f)
		{
			challengeableCount += 1.0f;
		}

		if (flatState[baseIndex + 1] > -9.5f)
		{
			knownEnemySpeedCount += 1.0f;
		}
	}

	state.values[0] = static_cast<float>(neighbors.size());
	state.values[1] = static_cast<float>(size);
	state.values[2] = challengeableCount;
	state.values[3] = knownEnemySpeedCount;
	state.values[4] = flatState[0]; // first-neighbor power snapshot
	state.values[5] = flatState[3]; // first-neighbor size snapshot
	state.values[6] = static_cast<float>(player.GetSpeed());
	state.values[7] = hasPlayed ? 1.0f : 0.0f;

	return state;
}

bool Tile::IsNeuralNetLoaded()
{
	return FileExists(kModelPath);
}

bool Tile::ChooseStayOnFloor()
{
	if (!IsNeuralNetLoaded())
	{
		return false;
	}

	StayGoState stayGoState = GetStayGoState();
	std::array<float, FLAT_STATE_SIZE> nnState{};
	nnState.fill(0.0f);

	for (int i = 0; i < STAYGO_STATE_SIZE; ++i)
	{
		nnState[i] = stayGoState.values[i];
	}

	const int action = PredictActionFromModel(nnState, 2);
	return action == 1;
}
