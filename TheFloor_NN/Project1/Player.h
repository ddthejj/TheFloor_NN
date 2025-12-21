#pragma once
#include <vector>

static constexpr int MAX_SKILL = 10;

class Player
{
	// original category (serves as unique identifier as well)
	int originalCategory;
	// skill level in each category
	std::vector<int> skillLevels;
	// overall speed at trivia
	int speed;
	// if this player has played before. If true, give the opponent knowledge of the opponent's speed when making decisions
	bool hasPlayed = false;

public:

	Player(int _originalCategory, int numCategories);

	int GetSkill(int category);
	int GetSpeed() { return speed; }
	int GetOriginalCategory() { return originalCategory; }
};

