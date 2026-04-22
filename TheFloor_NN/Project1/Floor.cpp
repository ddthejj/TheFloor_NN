#include "Floor.h"

#include "Player.h"
#include <iostream>
#include <algorithm>


Floor::Floor(int _floorWidth, int _floorHeight) : floorWidth(_floorWidth), floorHeight(_floorHeight), numCategories(_floorWidth * _floorHeight)
{
	floorTiles.resize(floorWidth);

	for (int x = 0; x < floorWidth; x++)
	{
		floorTiles[x].reserve(floorHeight);

		for (int y = 0; y < floorHeight; y++)
		{
			int category = x * floorHeight + y;

			floorTiles[x].emplace_back(category, numCategories);
			validRandomizerTiles.push_back(&floorTiles[x][y]);
			remainingTiles.push_back(&floorTiles[x][y]);
		}
	}

	// set up borders
	for (int x = 0; x < floorWidth; x++)
	{
		for (int y = 0; y < floorHeight; y++)
		{
			// add neighbors
			if (x > 0)
			{
				// west neighbor
				floorTiles[x][y].AddNeighbor(&floorTiles[x - 1][y]);
			}
			if (x < floorWidth - 1)
			{
				// east neighbor
				floorTiles[x][y].AddNeighbor(&floorTiles[x + 1][y]);
			}
			if (y > 0)
			{
				// north neighbor
				floorTiles[x][y].AddNeighbor(&floorTiles[x][y - 1]);
			}
			if (y < floorHeight - 1)
			{
				// south neighbor
				floorTiles[x][y].AddNeighbor(&floorTiles[x][y + 1]);
			}
		}
	}
}

int Floor::PlayGame()
{
	while (remainingTiles.size() > 1)
	{
		Step();
	}

	remainingTiles[0]->ResolveDecision(1.f, true);
	return remainingTiles[0]->GetPlayer()->GetOriginalCategory();
}

void Floor::Step()
{
	Tile* attackerTile = nullptr;

	if (!currentPlayer)
	{
		attackerTile = ActivateTheRandomizer();
	}
	else
	{
		attackerTile = currentPlayer;
	}

	// NEURAL NET OPTION
	// choose battle tile
	Tile* defenderTile = attackerTile->ChooseNeighbor();

	// do battle
	int category = defenderTile->GetCategory();

	// get the relative powers of the players
	// double the power means double the win odds
	int attackerPower = attackerTile->GetPower(category);
	int defenderPower = defenderTile->GetPower(category);
	// determine attacker win chance
	float attackerWinChance = (float)attackerPower / ((float)attackerPower + (float)defenderPower);
	// Generate winner
	double randomValue = static_cast<double>(std::rand()) / RAND_MAX;

	std::cout << "BATTLE " << currentBattle << ": " << attackerTile->GetPlayer()->GetOriginalCategory() << " VS " << defenderTile->GetPlayer()->GetOriginalCategory() << " IN " << category << '\n';

	Tile* winnerTile = nullptr;

	if (randomValue < attackerWinChance)
	{
		// attacker wins
 		winnerTile = attackerTile;

		attackerTile->WinBattle(true, defenderTile);
		defenderTile->LoseBattle();
		remainingTiles.erase(std::remove(remainingTiles.begin(), remainingTiles.end(), defenderTile), remainingTiles.end());
	}
	else
	{
		// defender wins
		winnerTile = defenderTile;

		defenderTile->WinBattle(false, attackerTile);
		attackerTile->LoseBattle();
		remainingTiles.erase(std::remove(remainingTiles.begin(), remainingTiles.end(), attackerTile), remainingTiles.end());
	}

	validRandomizerTiles.erase(std::remove(validRandomizerTiles.begin(), validRandomizerTiles.end(), attackerTile), validRandomizerTiles.end());
	validRandomizerTiles.erase(std::remove(validRandomizerTiles.begin(), validRandomizerTiles.end(), defenderTile), validRandomizerTiles.end());

	std::cout << "Winner: " << winnerTile->GetPlayer()->GetOriginalCategory() << '\n';
	
	// NEURAL NET OPTION
	// back to the floor?
	double newrandomValue = static_cast<double>(std::rand()) / RAND_MAX;
	if (newrandomValue > .5)
	{
		currentPlayer = winnerTile;
		std::cout << "STAY AND PLAY\n";

		attackerTile->ResolveDecision(.05f, false);
	}
	else
	{
		currentPlayer = nullptr;
		std::cout << "BACK TO THE FLOOR\n";

		attackerTile->ResolveDecision(-.5f, false);
	}

	currentBattle++;
}

Tile* Floor::ActivateTheRandomizer()
{
	if (validRandomizerTiles.size() > 0)
	{
		return validRandomizerTiles[(std::rand() % validRandomizerTiles.size())];
	}
	else
	{
		return remainingTiles[(std::rand() % remainingTiles.size())];
	}
}
