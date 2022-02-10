#include "BNN.hpp"

int main() {

	//Learns on 64 bit occupy & mask => 64 x 64 x 1 [per output bit]
	//BNN::FindSolution();

	//Learns on 16 bit pext => 16 x 32 x 1 [per output bit]
	BNN::FindSolutionPEXT();

	std::cout << "Copy paste the emitted code - recompile and press any key to verify solution!";
	std::ignore = std::getchar();

	BNN::CompareSolution();
}