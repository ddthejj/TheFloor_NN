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

public:

	Player(int _originalCategory, int numCategories);

	int GetSkill(int category);
	int GetSpeed() { return speed; }
	int GetOriginalCategory() { return originalCategory; }
};

