#ifndef CPURANDOM_H
#define CPURANDOM_H

#include <random>

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> dist_float(0.0f, 10.0f);
std::uniform_int_distribution<int> dist_int(0, 10);

#endif