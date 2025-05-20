#include "graph.h"
#include "utils.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <utility>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <cstring>
#include <stdlib.h>
#include <cstdlib>


using namespace std;

Graph::Graph(string routes_filepath, string airports_filepath) {
    createAirportsMap(airports_filepath);
    createAdjacencyList(routes_filepath);
}

Graph::~Graph() {   }

Graph& Graph::operator=(const Graph& g) {
    adjacency_list_ = g.adjacency_list_;
    return *(this);
}

// Method from
// https://stackoverflow.com/questions/14265581/parse-split-a-string-in-c-using-string-delimiter-standard-c
vector<string> Graph::split (string s, string delimiter) {
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    string token;
    vector<string> res;

    while ((pos_end = s.find (delimiter, pos_start)) != string::npos) {
        token = s.substr (pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back (token);
    }

    res.push_back (s.substr (pos_start));
    return res;
}

void Graph::createAirportsMap(string filepath) {
    ifstream infile(filepath);

    if (!infile || infile.peek() == ifstream::traits_type::eof()) {
        throw std::invalid_argument( "Parsing error 1" );
    }

    string line;
    string delimiter = ",";
    while (getline(infile, line)) {
        vector<string> v = split(line, delimiter);

        Node node;
        node.id_ = stoi(v[0]);
        node.IATA_ = v[1];
        node.ICAO_ = v[2];
        node.latitude_ = stod(v[3]);
        node.longitude_ = stod(v[4]);

        airports_map_.insert({node.id_, node});
        addNode(node);

        // Node n = airports_map_.at(node.id_);
        // std::cout << "ID: " << n.id_ << ", Latitude: " << n.latitude_ << ", Longitude: " << n.longitude_ << std::endl;
    }

    // fix bug for \N (0) airport ID
    Node node;
    node.id_ = 0;
    node.IATA_ = "NaN";
    node.ICAO_ = "NaN";
    node.latitude_ = 0;
    node.longitude_ = 0;

    airports_map_.insert({node.id_, node});
    addNode(node);
}

void Graph::createAdjacencyList(string filepath) {
    ifstream infile(filepath);

    if (!infile || infile.peek() == ifstream::traits_type::eof()) {
        throw std::invalid_argument( "Parsing error 2" );
    }

    string line;
    string delimiter = ",";
    while (getline(infile, line)) {
        size_t pos = line.find(delimiter);

        if (pos == string::npos) {
            throw std::invalid_argument( "Malformed data" );
        }

        int source = stoi(line.substr(0, line.find(delimiter)));
        // if (source == '\n') {
        //     source = "0";
        // }

        int dest = stoi(line.substr(line.find(delimiter) + 1, line.length()));
        // if (dest == '\n') {
        //     dest = "0";
        // }
        
        // check if valid airport b/c airports.dat did not contain all airport IDs in routes.dat (ex. ID: 7147)
        if (airports_map_.count(source) && airports_map_.count(dest)) {
            // cout << source << " : " << dest << endl;
            // cout << "Test 1" << endl;
            Node source_node = airports_map_.at(source);
            // cout << "Test 2" << endl;
            Node dest_node = airports_map_.at(dest);
            // cout << "Test 3" << endl;
            addEdge(source_node, dest_node);
        }
        
    }
}

void Graph::addNode(Node node) {
    adjacency_list_.insert({node, vector<Edge>(0)});
}

void Graph::addEdge(Node source, Node dest) {
    // cout << "Test 4 : " << source.id_ << endl;
    vector<Edge> & edges = adjacency_list_.at(source);

    // check for duplicates
    for (Edge edge : edges) {
        if (edge.dest_.id_ == dest.id_) {
            return;
        }
    }

    Edge edge;
    edge.source_ = source;
    edge.dest_ = dest;
    edge.weight_ = getWeight(source, dest);
    
    edges.push_back(edge);
}

double Graph::getWeight(Node start, Node end) {
    double source_lat = start.latitude_;
    double source_long = start.longitude_;
    double dest_lat = end.latitude_;
    double dest_long = end.longitude_;

    return calculateDistance(source_lat, source_long, dest_lat, dest_long);
}

map<Graph::Node, vector<Graph::Edge>> & Graph::getAdjacencyList() {
    return adjacency_list_;
}

vector<Graph::Node> Graph::getAirports() {
    std::vector<Node> airports;
    for(map<int, Node>::iterator it = airports_map_.begin(); it != airports_map_.end(); ++it) {
        airports.push_back(it->second);
    }

    return airports;
}

map<int, Graph::Node> & Graph::getAirportsMap() {
    return airports_map_;
}

string Graph::getAirportLabel(Graph::Node node) {
    string label = "";
    if (node.IATA_ != "NaN") {
        label = node.IATA_;
    } else if (node.ICAO_ != "NaN") {
        label = node.ICAO_;
    } else {
        label = to_string(node.id_);
    }

    return label;
}

void Graph::printGraph(std::ostream& output) {
    for (auto pair : adjacency_list_) {
        string main_label = getAirportLabel(pair.first);

        output << "Airport ID - " << main_label << ": ";
        for (auto it = pair.second.begin(); it != pair.second.end(); ++it) {
            string label = getAirportLabel(it->dest_);
            if (std::next(it) != pair.second.end()) {
                output << label << " ";
            } else {
                output << label;
            }
        }
        output << endl;
    }
}

Graph::Graph(const Graph &g) {
    adjacency_list_ = g.adjacency_list_;
}

int Graph::getSize() {
    return adjacency_list_.size();
}

Graph::Node Graph::labelToId(string label) { 
    toUpper(label);
    for (auto pair : airports_map_) {
        if (pair.second.IATA_ == label || pair.second.ICAO_ == label) {
            return pair.second;
        }
    }

    Graph::Node defaultNode {0, "NaN", "NaN", 0, 0};
    return defaultNode;
}
