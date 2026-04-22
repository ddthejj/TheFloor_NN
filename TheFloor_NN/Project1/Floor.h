#pragma once

#include <vector>
#include "Tile.h"


class Floor
{
	// size of the floor
	int floorWidth;
	int floorHeight;
	// how many categories are remaining on the floor
	int numCategories;
	// 2d array of tiles
	std::vector<std::vector<Tile>> floorTiles;

	std::vector<Tile*> validRandomizerTiles;
	std::vector<Tile*> remainingTiles;

	Tile* currentPlayer;
	int currentBattle = 0;

public:

	Floor(int _floorWidth, int _floorHeight);

	int PlayGame();
	void Step();

	Tile* ActivateTheRandomizer();
};

