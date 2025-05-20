#include "utils.h"

#define PI 3.14159265358979323846

// Distance calculation based on
// https://stackoverflow.com/questions/27126714/c-latitude-and-longitude-distance-calculator

double radian(double degree) {
    return (degree / 180) * PI;
}

double calculateDistance(double source_lat, double source_long, double dest_lat, double dest_long) {
    double dist;
    dist = sin(radian(source_lat)) * sin(radian(dest_lat)) + cos(radian(source_lat)) * cos(radian(dest_lat)) * cos(radian(source_long - dest_long));
    dist = acos(dist) * 6371;

    return dist;
}

void toUpper(string& str) {
    for (auto & c: str) {
       c = toupper(c); 
    } 
}