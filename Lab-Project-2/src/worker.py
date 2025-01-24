import sys

import matplotlib.pyplot as plt
import networkx as nx

from utility import *
from solvers import *
from visualizer import visualize_path

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <algorithm> [num_nodes] [sparsity]")
    else:
        algo = sys.argv[1].lower()

        # Get optional num_nodes and sparsity arguments
        num_nodes = 20  # Default values
        sparsity = 0.03
        if len(sys.argv) > 2:
            try:
                num_nodes = int(sys.argv[2])
                sparsity = float(sys.argv[3])
                if num_nodes <= 0:
                    raise ValueError
            except ValueError:
                print("Invalid number of nodes or sparsity provided. Using default values.")

        G = generate_random_graph(num_nodes, sparsity)
        start, finish = select_start_finish(G)

        if algo == "ucs":
            path, cost, G = uniform_cost_search(G, start, finish)
        elif algo == "dfs":
            path, cost, G = depth_first_search(G, start, finish)
        elif algo == "bfs":
            path, cost, G = breadth_first_search(G, start, finish)
        else:
            print(f"Invalid algorithm: {algo}. Choose from ucs, dfs, or bfs.")
            sys.exit(1)

        visualize_path((algo).upper(), G, path, cost)

if __name__ == "__main__":
    main()