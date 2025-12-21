#include "Floor.h"
#include <iostream>
#include <cstdlib>
#include <ctime>

int main(int argc, char* argv[]) 
{
	// seed random with current time
	std::srand(std::time({}));

	// create the floor
	Floor TheFloor(10, 10);
	std::cout << "THE FLOOR WINNER:" << TheFloor.PlayGame();
}	