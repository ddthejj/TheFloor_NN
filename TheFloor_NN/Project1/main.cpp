#include "Floor.h"
#include <iostream>
#include <cstdlib>
#include <ctime>

static constexpr int TEST_GAMES = 1;

int main(int argc, char* argv[])
{
	// seed random with current time
	std::srand(std::time({}));

	// create the floor

	for (int i = 0; i < TEST_GAMES; i++)
	{
		Floor TheFloor(10, 10);
		std::cout << "THE FLOOR WINNER:" << TheFloor.PlayGame() << "\n";
		std::cout << "GAME " << i << " COMPLETE\n"; 
	}
}