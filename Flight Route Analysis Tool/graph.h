#pragma once

#include <map>
#include <vector>
#include <string>
#include <algorithm>

using namespace std;

class Graph {
    public:
        struct Node {
            int id_;
            string IATA_;
            string ICAO_;
            double latitude_;
            double longitude_;

            bool operator< (const Node& other) const {
                return id_ < other.id_;
            }

            bool operator> (const Node& other) const {
                return id_ > other.id_;
            }

            bool operator== (const Node& other) const {
                return id_ == other.id_;
                        // && latitude_ == other.latitude_
                        // && longitude_ == other.longitude_;
            }

            bool operator!= (const Node& other) const {
                return id_ != other.id_;
                        // && latitude_ == other.latitude_
                        // && longitude_ == other.longitude_;
            }
        };

        struct Edge {
            Node source_;
            Node dest_;
            double weight_;

            bool operator== (const Edge& other) const {
                return source_.id_ == other.source_.id_ && dest_.id_ == other.dest_.id_;
            }
        };

        Graph(string routes_filepath, string airports_filepath);
        
        ~Graph();

        Graph(const Graph &g);

        Graph& operator=(const Graph& g);

        void printGraph(std::ostream& output);

        vector<string> split(string s, string delimiter);

        void createAirportsMap(string filepath);

        void createAdjacencyList(string filepath);

        map<Node, vector<Edge>> & getAdjacencyList();

        map<int, Node> & getAirportsMap();

        vector<Node> getAirports();

        double getWeight(Node start, Node end);

        int getSize();

        string getAirportLabel(Graph::Node node);

        Node labelToId(string label);

    private:
        map<Node, vector<Edge>> adjacency_list_;

        map<int, Node> airports_map_;

        void addNode(Node node);

        void addEdge(Node start, Node end);
};
