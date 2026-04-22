#include "Tile.h"

#include <cstdlib>

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

	// NEURAL NET OPTION
	
	//return neighbors[(std::rand() % neighbors.size())];

	MLState state = GetMLState();

	int action = (std::rand() % neighbors.size());

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

		neighborState.myPower = GetPower(neighbor->GetCategory());
		neighborState.enemySpeed = neighbor->hasPlayed ? neighbor->GetPlayer()->GetSpeed() : -10;
		neighborState.mySize = size;
		neighborState.enemySize = neighbor->size;

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
