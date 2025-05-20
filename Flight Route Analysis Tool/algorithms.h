#pragma once

#include "graph.h"

#include <vector>
#include <stack>
#include <limits>
#include <queue>
#include <set>
#include <unordered_map>

using namespace std;

vector<Graph::Node> BFSHelper(Graph::Node starting_node, map<Graph::Node, vector<Graph::Edge>>& adjacency_list, map<Graph::Node, bool>& visited);

vector<vector<Graph::Node>> BFS(Graph::Node starting_node, Graph* graph);

vector<Graph::Node> shortestPath(Graph* graph, Graph::Node starting_node, Graph::Node dest_node);

vector<Graph::Node> PageRank(Graph* graph, int num_to_print);

vector<Graph::Node> PageRank(Graph* graph);
