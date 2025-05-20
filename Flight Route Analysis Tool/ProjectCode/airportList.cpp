#include "airportList.h"
#include <string>
#include <vector>
#include <iostream>
#include <fstream>

using namespace std;

traverse(){
    ifstream fin("airports.dat"); 
    file.open("airports.dat");

    cout << 

    int num;
    vector<string> vec;

}


// default constructor 

AirportList::AirportInitial(){

    airID = 0;
    airName = "";
    airCity = "";
    airCountry = "";
    airLat = 0.0;
    airLong = 0.0;

}



//constructor 

AirportList:: AirportInitial(int id, string name, string city, string country, double latitude, double longitude):
(airID(id), airName(name), airCity(city), airCountry(country), airLat(latitude), airLong(longitude))
{ 
  
}

// getters
int AirportList::getAirID{

    return airID;
}

int AirportList::getAirName{
    return airName;
}

int AirportList::getAirCity{
    return airCity;
}

int AirportList::getAirCountry{
    return airCountry;
}

int AirportList::GetAirLat{

    return airLat;
}

int AirportList::getAirLong{
    return airLong;
}
