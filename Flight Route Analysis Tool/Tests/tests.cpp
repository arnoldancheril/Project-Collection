#define CATCH_CONFIG_MAIN

#include "catch.hpp"
#include "../algorithms.h"
#include "../graph.h"
#include <sstream>
#include <cassert>
#include <stdexcept>

using namespace std;

// BFS Tests
TEST_CASE("BFS on connected graph", "[bfs]") {
    Graph*  g = new Graph("./Tests/test_data/test_routes_small_data.txt", "./Tests/test_data/test_airports_small_data.txt");

    Graph::Node n0;
    n0.id_ = 0;
    Graph::Node n1;
    n1.id_ = 1;
    Graph::Node n2;
    n2.id_ = 2;
    Graph::Node n3;
    n3.id_ = 3;
    Graph::Node n4;
    n4.id_ = 4;

    vector<vector<Graph::Node>> expected{{n0, n2, n4, n3, n1}};

    Graph::Node starting_node;
    starting_node.id_ = 0;

    REQUIRE(expected == BFS(starting_node, g));
}

TEST_CASE("BFS on disconnected graph", "[bfs]") {
    Graph* g = new Graph("./Tests/test_data/test_routes_small_disconnected_data.txt", "./Tests/test_data/test_airports_small_disconnected_data.txt");

    Graph::Node n0;
    n0.id_ = 0;
    Graph::Node n1;
    n1.id_ = 1;
    Graph::Node n2;
    n2.id_ = 2;
    Graph::Node n3;
    n3.id_ = 3;
    Graph::Node n4;
    n4.id_ = 4;
    Graph::Node n5;
    n5.id_ = 5;
    Graph::Node n6;
    n6.id_ = 6;

    vector<vector<Graph::Node>> expected{{n0, n2, n4, n3, n1}, {n5, n6}};

    Graph::Node starting_node;
    starting_node.id_ = 0;

    REQUIRE(expected == BFS(starting_node, g));
}

TEST_CASE("BFS proper traversal on a graph with a single node", "[bfs]") {
    Graph* g = new Graph("./Tests/test_data/test_routes_single_data.txt", "./Tests/test_data/test_airports_single_data.txt");

    Graph::Node n0;
    n0.id_ = 0;

    vector<vector<Graph::Node>> expected{{n0}};

    Graph::Node starting_node;
    starting_node.id_ = 0;

    REQUIRE(expected == BFS(starting_node, g));
}

// Dijkstras Tests
TEST_CASE("Dijkstras works properly on a direct flight", "[dijkstra]") {
    Graph* g = new Graph("OpenFlightsDataset/routes_parsed.txt", "./Tests/test_data/test_dijkstras_small_data.txt");

    Graph::Node n0;
    n0.id_ = 4049;
    Graph::Node n1;
    n1.id_ = 3830;

    vector<Graph::Node> expected{n0, n1};

    Graph::Node starting_node {4049,"CMI","KCMI",40.03919983,-88.27809906};
    Graph::Node dest_node {3830,"ORD","KORD",41.9786,-87.9048};


    REQUIRE(expected == shortestPath(g, starting_node, dest_node));
}

TEST_CASE("Dijkstras works properly on a flight with multiple stops", "[dijkstra]") {
    Graph* g = new Graph("OpenFlightsDataset/routes_parsed.txt", "./Tests/test_data/test_dijkstras_small_data.txt");

    Graph::Node n0;
    n0.id_ = 1;
    Graph::Node n1;
    n1.id_ = 5;
    Graph::Node n2;
    n2.id_ = 1960;
    Graph::Node n3;
    n3.id_ = 3484;
    Graph::Node n4;
    n4.id_ = 3830;

    vector<Graph::Node> expected{n0, n1, n2, n3, n4};

    Graph::Node starting_node {1,"GKA","AYGA",-6.081689834590001,145.391998291};
    Graph::Node dest_node {3830,"ORD","KORD",41.9786,-87.9048};

    REQUIRE(expected == shortestPath(g, starting_node, dest_node));
}

TEST_CASE("Checking PageRank Algorithm is able to print the top 100 ranked airports of the datasets"){
    Graph* g = new Graph("./OpenFlightsDataset/routes_parsed.txt", "./OpenFlightsDataset/airports_parsed.txt");

    PageRank(g);
}
