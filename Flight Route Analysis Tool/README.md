# Flight Route Analysis Tool

## Project Overview
This project is a flight route analysis tool that uses data from OpenFlights.org to analyze airport connections and find optimal routes between airports. The tool provides various algorithms to analyze flight routes:

1. **Breadth-First Search (BFS)** - Traverses the flight network starting from a specified airport
2. **Dijkstra's Algorithm** - Finds the shortest path between two airports
3. **PageRank Algorithm** - Ranks airports based on their importance in the flight network

## Project Directory Structure
```
.
├── OpenFlightsDataset/        # Processed data from OpenFlights.org
├── Tests/                     # Test suite for algorithms
│   ├── catch.hpp              # Catch2 test framework
│   ├── test_data/             # Test datasets
│   └── tests.cpp              # Test implementation
├── Outputs/                   # Directory for algorithm outputs
├── algorithms.cpp             # Implementation of BFS, Dijkstra's, and PageRank
├── algorithms.h               # Header file for algorithms
├── airports.csv               # Raw airport data
├── airports_parser.py         # Python script for parsing airport data
├── graph.cpp                  # Graph implementation using adjacency list
├── graph.h                    # Header file for graph class
├── main.cpp                   # Main program with CLI interface
├── main                       # Compiled executable
├── Makefile                   # Build instructions
├── results.md                 # Written report of findings
├── routes.csv                 # Raw routes data
├── routes_parser.py           # Python script for parsing routes data
├── utils.cpp                  # Utility functions
└── utils.h                    # Header file for utilities
```

## Data
The project uses two main datasets from OpenFlights.org:
- **airports.csv**: Contains information about airports including ID, name, IATA code, city, country, latitude, longitude
- **routes.csv**: Contains information about flight routes between airports

The raw data is processed using Python scripts (`airports_parser.py` and `routes_parser.py`) to convert them into a usable format for the graph implementation.

## Implementation Details

### Graph Representation
- Airports are represented as nodes in the graph
- Flight routes are represented as edges
- The distance between airports is calculated using geographical coordinates and used as edge weights
- An adjacency list is used to store the graph structure

### Algorithms

#### 1. Breadth-First Search (BFS)
- Traverses the airport network starting from a given airport
- Visits all connected airports before moving to the next level
- Identifies separate connected components in the graph
- Time Complexity: O(V + E) where V is the number of airports and E is the number of routes

#### 2. Dijkstra's Algorithm
- Finds the shortest path between two airports based on distance
- Uses a priority queue to efficiently select the next closest airport
- Returns the sequence of airports that forms the shortest path
- Time Complexity: O(E log V) where V is the number of airports and E is the number of routes

#### 3. PageRank Algorithm
- Ranks airports based on their importance in the flight network
- Importance is determined by the number of incoming and outgoing flights
- Uses a damping factor of 0.85 (standard for PageRank)
- Time Complexity: O(V + E) per iteration, where multiple iterations are performed

## Setup and Build Instructions

### Prerequisites
- C++ compiler with C++14 support (clang++ recommended)
- Make

### Building the Project
To build the project:
```
make
```

This will compile all the necessary files and create the executable `main`.

## Running Instructions

### Running Tests
To build and run the test suite:
```
make test
./test
```

### Running BFS Algorithm
To run a BFS traversal starting from an airport:
```
./main FRA g bfs
```
Where:
- `FRA` is the IATA code of the starting airport
- `g` is the graph identifier
- `bfs` indicates to run the BFS algorithm

### Running Dijkstra's Algorithm
To find the shortest path between two airports:
```
./main FRA PHX dij
```
Where:
- `FRA` is the IATA code of the starting airport
- `PHX` is the IATA code of the destination airport
- `dij` indicates to run Dijkstra's algorithm

### Running PageRank Algorithm
To rank airports based on their importance:
```
./main Page
```
After running this command, you'll be prompted to enter the number of airports you want to rank (up to 7699).

## Test Suite
The test suite includes:
- 3 tests for BFS algorithm (connected graph, disconnected graph, single node)
- 2 tests for Dijkstra's algorithm (direct flight, multiple stops)
- 1 test for PageRank algorithm (ranking top 100 airports)

## Results and Findings
The results of our analysis are documented in `results.md`, which includes:
- Description of the data processing methodology
- Implementation details of each algorithm
- Analysis of the results and their practical applications
- Answers to our leading question about finding the shortest path between airports

## Contributors
This project was developed as part of CS 225 coursework.

## Resources
- [OpenFlights.org](https://openflights.org/data.html) - Source of airport and route data
- [Breadth-First Search](https://en.wikipedia.org/wiki/Breadth-first_search)
- [Dijkstra's Algorithm](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm)
- [PageRank Algorithm](https://en.wikipedia.org/wiki/PageRank)
