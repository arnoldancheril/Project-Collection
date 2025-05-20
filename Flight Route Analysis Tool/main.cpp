#include "graph.h"
#include "utils.h"
#include "algorithms.h"
#include <iostream>
#include <fstream>

using namespace std;

int main(int argc, const char** argv) {
    Graph* g = new Graph("./OpenFlightsDataset/routes_parsed.txt", "./OpenFlightsDataset/airports_parsed.txt");
    // Graph* g = new Graph("./Tests/test_data/test_routes_small_data.txt", "./Tests/test_data/test_airports_small_data.txt");

    // g->printGraph(cout);
    string tempCheck = argv[1];
    if (tempCheck == "Page") {    
        int num_to_print; 
        cout << "How many airports would you like to have ranked? (Up to 7699)" << endl;
        cin >> num_to_print;
        cout << endl;
        
        vector<Graph::Node> pageRank = PageRank(g, num_to_print);

        return 0;
    }

    Graph::Node starting_node = g->labelToId(argv[1]);
    Graph::Node dest_node = g->labelToId(argv[2]);
    string opt = argv[3];
    toUpper(opt);

    // Graph::Node starting_node {
    //     1,
    //     "GKA",
    //     "AYGA",
    //     -6.081689834590001,
    //     145.391998291
    // };

    // Graph::Node dest_node {
    //     3830,
    //     "ORD",
    //     "KORD",
    //     41.9786,
    //     -87.9048
    // };

    if (opt == "BFS") {
        // BFS Traversal
        cout << endl;
        cout << "BFS Traversal:" << endl;
        cout << "========================================================" << endl;
        // Graph::Node starting_node;
        // starting_node.id_ = 1;
        // starting_node.latitude_ = -6.081689834590001;
        // starting_node.longitude_ = 145.391998291;
        vector<vector<Graph::Node>> bfs = BFS(starting_node, g);    
        int count = 0;
        for (size_t i = 0; i < bfs.size(); i++) {
            if (count < 1) {
                cout << "Component " << i + 1 << ": ";
                for (size_t j = 0; j < bfs[i].size(); j++) {
                    string label = g->getAirportLabel(bfs[i][j]);
                    if (j == bfs[i].size() - 1) {
                        cout << label << endl;
                    } else {
                        cout << label << "-->";
                    }
                }
                cout << endl;

                count++;
            }
            
        }
    }
    
    if (opt == "DIJ") {
        vector<Graph::Node> path = shortestPath(g, starting_node, dest_node);
        cout << "Shortest Path between " << g->getAirportLabel(starting_node) << " and " << g->getAirportLabel(dest_node) << endl;
        cout << "=================================================================" << endl;
        if (path.empty()) {
            cout << "No path" << endl;
        } else {
            for (size_t i = 0; i < path.size(); i++) {
                string label = g->getAirportLabel(path[i]);
                if (i != path.size() - 1) {
                    cout << label << " -> "; 
                } else {
                    cout << label;
                }
                
            }
            cout << endl;
        }
    }

    return 0;
}
