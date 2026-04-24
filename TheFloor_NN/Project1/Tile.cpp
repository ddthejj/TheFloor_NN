#include "Tile.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

#ifdef _WIN32
#define POPEN _popen
#define PCLOSE _pclose
#else
#define POPEN popen
#define PCLOSE pclose
#endif

namespace
{
	constexpr const char* kPredictScriptPath = "NN_Training/predict_action.py";
	constexpr const char* kModelPath = "NN_Training/artifacts/model.keras";
	constexpr const char* kNormPath = "NN_Training/artifacts/norm.json";

	bool FileExists(const char* filePath)
	{
		std::ifstream file(filePath);
		return file.good();
	}

	int PredictActionFromModel(const std::array<float, FLAT_STATE_SIZE>& flatState, int validNeighborCount)
	{
		if (!FileExists(kModelPath))
		{
			return -1;
		}
		const bool hasNormFile = FileExists(kNormPath);

		std::ostringstream stateBuilder;
		for (int i = 0; i < FLAT_STATE_SIZE; ++i)
		{
			if (i > 0)
			{
				stateBuilder << ",";
			}

			stateBuilder << flatState[i];
		}

		std::ostringstream commandBuilder;
		commandBuilder << "printf '%s' '" << stateBuilder.str() << "'"
			<< " | python \"" << kPredictScriptPath << "\""
			<< " --model \"" << kModelPath << "\"";
		if (hasNormFile)
		{
			commandBuilder << " --norm \"" << kNormPath << "\"";
		}

		commandBuilder
			<< " --valid-count " << validNeighborCount;

		FILE* process = POPEN(commandBuilder.str().c_str(), "r");
		if (process == nullptr)
		{
			return -1;
		}

		char outputBuffer[64] = {};
		if (std::fgets(outputBuffer, sizeof(outputBuffer), process) == nullptr)
		{
			PCLOSE(process);
			return -1;
		}

		PCLOSE(process);
		return std::atoi(outputBuffer);
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

	

	return state;
}
