#include "Player.h"

#include <cstdlib>

Player::Player(int _originalCategory, int numCategories) : originalCategory(_originalCategory), skillLevels(numCategories)
{
	// create and set random skill levels
	for (int i = 0; i < numCategories; i++)
	{
		skillLevels[i] = (std::rand() % MAX_SKILL) + 1;
	}

	// set random trivia speed
	speed = (std::rand() % MAX_SKILL) + 1;

	// set our original category skill level to the highest value
	skillLevels[_originalCategory] = MAX_SKILL;
}

int Player::GetSkill(int category)
{
	return skillLevels[category];
}
