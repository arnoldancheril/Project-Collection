#pragma once

#include <cmath>
#include <string>

using namespace std;

double radian(double degree);

double calculateDistance(double source_lat, double source_long, double dest_lat, double dest_long);

void toUpper(string& str);