#pragma once

#include "Player.h"
#include <vector>

class Tile
{
	int category;
	Player player;

	std::vector<Tile*> neighbors;

	bool hasLost = false;

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
};