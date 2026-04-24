#pragma once

#include "Player.h"
#include <vector>
#include "MLState.h"

class Tile
{
	int category;
	Player player;

	std::vector<Tile*> neighbors;

	bool hasLost = false;
	// if this tile has been played before. If true, give the opponent knowledge of the this player's speed when making decisions
	bool hasPlayed = false;

	int size = 1;

	MLState lastState;
	int lastAction = -1;
	bool hasPendingDecision = false;

public:

	Tile(int _category, int numCategories);

	int GetCategory() { return category; }
	int GetPower(int battleCategory);
	Player* GetPlayer() { return &player; }

	void AddNeighbor(Tile* neighbor);
	void RemoveNeighbor(Tile* neighbor);
	Tile* ChooseNeighbor();

	void WinBattle(bool wasAttacker, Tile* loserTile);
	void LoseBattle();

	MLState GetMLState();
	void StoreDecision(MLState& state, int action);
	bool HasPendingDecition() { return hasPendingDecision; }
	void ResolveDecision(float reward, bool done);
	StayGoState GetStayGoState();
	bool IsNeuralNetLoaded();
	bool ChooseStayOnFloor();
};
