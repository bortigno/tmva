
#include <fstream>
#include <iostream>
#include <vector>
#include "../pattern/pattern.hpp"

class Pattern;

int reverseInt (int i);
std::vector<Pattern> read_Mnist(std::string filenameImages, std::string filenameLabels, int maxImages = -1);


