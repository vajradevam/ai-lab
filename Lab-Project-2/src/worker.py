import sys

import matplotlib.pyplot as plt
import networkx as nx

from solvers import uniform_cost_search
from utility import *
from visualizer import visualize_ucs_path

# G = read_undirected_graph("./test.gph")
G = generate_random_graph(20)
start, finish = select_start_finish(G)
path, cost, G = uniform_cost_search(G, start, finish)
visualize_ucs_path(G, path, cost)