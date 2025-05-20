#include "algorithms.h"
#include <map>
#include <queue>

#include <iostream>
#include <algorithm>

using namespace std;

vector<Graph::Node> BFSHelper(Graph::Node starting_node, map<Graph::Node, vector<Graph::Graph::Edge>>& adjacency_list, map<Graph::Node, bool>& visited) {
    vector<Graph::Node> result;
    queue<Graph::Node> q;

    q.push(starting_node);
    visited.at(starting_node) = true;
    
    while (!q.empty()) {
        Graph::Node cur = q.front();
        // size_t airport = cur.id_;
        q.pop();
        result.push_back(cur);

        for (size_t i = 0; i < adjacency_list.at(cur).size(); i++) {
            if (!visited.at(adjacency_list.at(cur)[i].dest_)) {
                visited.at(adjacency_list.at(cur)[i].dest_) = true;
                q.push(adjacency_list.at(cur)[i].dest_);
            }
        }
    }

    return result;
}

vector<vector<Graph::Node>> BFS(Graph::Node starting_node, Graph* graph) {
    map<Graph::Node, bool> visited;
    map<Graph::Node, vector<Graph::Graph::Edge>> adjacency_list = graph->getAdjacencyList();

    for (auto it = adjacency_list.begin(); it != adjacency_list.end(); ++it) {
        visited.insert({it->first, false});
    }

    vector<vector<Graph::Node>> result;
    result.push_back(BFSHelper(starting_node, adjacency_list, visited));

    for (auto it = visited.begin(); it != visited.end(); ++it) {
        if (!(it->second)) {
            result.push_back(BFSHelper(it->first, adjacency_list, visited));
        }
    }
    
    return result;
}

// dijkstras to find shortest path between source and dest
vector<Graph::Node> shortestPath(Graph* graph, Graph::Node starting_node, Graph::Node dest_node) {
    map<Graph::Node, vector<Graph::Graph::Edge>> adjacency_list = graph->getAdjacencyList();
    vector<Graph::Edge> path;

    vector<Graph::Node> airports = graph->getAirports();
    map<Graph::Node, double> dist;
    map<Graph::Node, bool> visited;
    map<Graph::Node, Graph::Node> prev;
    
    for (size_t i = 0; i < airports.size(); i++) {
        Graph::Node cur_airport = airports[i];
        if (cur_airport == starting_node) {
            dist[cur_airport] = 0;
        } else {
            dist[cur_airport] = INT_MAX;
        }

        visited[cur_airport] = false;
    }

    
    // min heap priorty queue
    priority_queue<pair<double, Graph::Node>, vector<pair<double, Graph::Node>>, greater<pair<double, Graph::Node>>> pq;
    pq.push(pair<double, Graph::Node>(0, starting_node));

    while (pq.top().second != dest_node) {
        Graph::Node cur_node = pq.top().second;
        pq.pop();

        vector<Graph::Edge> neighbors = adjacency_list.at(cur_node);
        for (size_t i = 0; i < neighbors.size(); i++) {
            if (!visited[neighbors[i].dest_]) {
                Graph::Node cur_neighbor = neighbors[i].dest_;
                double new_weight = dist[cur_node] + graph->getWeight(cur_node, cur_neighbor);
                if (dist[cur_neighbor] > new_weight) {
                    dist[cur_neighbor] = new_weight;
                    prev[cur_neighbor] = cur_node;
                    pq.push(pair<double, Graph::Node>(dist[cur_neighbor], cur_neighbor));
                }
            }   
        }

        visited[cur_node] = true;
    }

    Graph::Node curr = dest_node;
    Graph::Node currprev = prev[curr];

    double weight = 0;
    for (curr = dest_node; curr != starting_node; curr = prev[curr]) {
        weight = graph->getWeight(prev[curr], curr);
        vector<Graph::Edge>::iterator it = path.begin();

        Graph::Edge temp_edge = {
            prev[curr],
            curr,
            weight
        };

        it = path.insert(it, temp_edge);
    }

    vector<Graph::Node> node_path;

    for (size_t i = 0; i < path.size(); i++) {
        node_path.push_back(path[i].source_);
        if (i == path.size() - 1) {
            node_path.push_back(path[i].dest_);
        }
    }

    return node_path;
}

//copy used for test cases, with ranking top 100 airports
vector<Graph::Node> PageRank(Graph* graph){
    return PageRank(graph, 100);
}

//https://en.wikipedia.org/wiki/PageRank#Implementation
vector<Graph::Node> PageRank(Graph* graph, int num_to_print){
    map<Graph::Node, vector<Graph::Graph::Edge>> adjacency_list = graph->getAdjacencyList();
    vector<pair<Graph::Node, long double>> node_to_int;
    vector<Graph::Node> result;

    //  going through the nodes of the adjacency list, checking vector size to determine airport connections
    int max_length = 0;
    string node = "";
    // int zerocounter = 0;
    for(auto kv: adjacency_list) {
        int len = kv.second.size();
        if (len > 0){
            
            //PageRank algorithm calculations
            long double pagerankValue = 0;
            long double initialPRValue = (1.0/7699.0);  //value of each node, assuming that airport only has one connection
            //cout << " Initial PageRank Value assuming each airport only has one connection: "<<initialPRValue << endl;
            long double linkAdjustmentTemp = 0.0;
            double dampingFactor = 0.85;  /*Damping factor in PageRank algorithm is the probability that someone will continue on from the node
                                            and go on to another node. On average the damping factor for the page rank algorithm is 0.85, 
                                            meaning that around 15% of people stay at that link/node and do not continue.*/
                                            //https://en.wikipedia.org/wiki/PageRank#Damping_factor
            linkAdjustmentTemp = ((dampingFactor) * (((initialPRValue) * (len))/ 100));
            long double correction = ((1.0 - dampingFactor) / 7699.0);
            //cout << " Correction is " << correction << endl;
            pagerankValue = linkAdjustmentTemp + correction;
            // cout << "PageRank Value is " << pagerankValue << endl;

            node_to_int.push_back(make_pair(kv.first,pagerankValue));
            // if(len >= max_length) {
            //     max_length = len;
            //     if(kv.first.IATA_ != "NaN"){
            //         node = kv.first.IATA_;
            //     }
            //     else if(kv.first.IATA_ == "NaN"){
            //         node = kv.first.ICAO_;
            //     }
            // }            
        }
        else{
            node_to_int.push_back(make_pair(kv.first,0));
        }
    }



    //sorting the vector node to int, by the second value of the pair which is in the int
    //https://stackoverflow.com/questions/5056645/sorting-stdmap-using-value
    //[=] part of lambada expression
    sort(node_to_int.begin(), node_to_int.end(), [=](std::pair<Graph::Node, long double>& x, std::pair<Graph::Node, long double>& y){
        return x.second > y.second;
    });

    
    //just going through and printing the values
    // int num_to_print; 
    // cout << "How many airports would you like to have ranked? (Up to 7699)" << endl;
    // cin >> num_to_print;
    // cout << endl;

    //int num_to_print = 100;  //total = 7699  //can use this instead of user input, to print top 100 most important airports
    for(int i = 0; i < num_to_print; i++) {
        if(node_to_int[i].first.IATA_ != "NaN") {
            cout << i+1 << ": " << node_to_int[i].first.IATA_ << ", Page Rank: " << node_to_int[i].second << endl;
        }
        else if(node_to_int[i].first.IATA_ == "NaN"){
            cout << i+1 << ": " << node_to_int[i].first.ICAO_ << ", Page Rank: " << node_to_int[i].second << endl;
        }
    }

    cout << endl;
    if(node_to_int[0].first.IATA_ != "NaN"){
        cout << "Most Important Airport: " << node_to_int[0].first.IATA_ << ", Page Rank Value is: " << node_to_int[0].second << endl;
    }
    else if(node_to_int[0].first.IATA_ == "NaN"){
        cout << "Most Important Airport: " << node_to_int[0].first.ICAO_ << ", Page Rank Value is: " << node_to_int[0].second << endl;
    }
    cout << endl;

    return result; 
}
