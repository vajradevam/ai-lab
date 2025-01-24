import sys
import random
import networkx as nx
import matplotlib.pyplot as plt

import networkx as nx
import random

def generate_random_graph(n, sparsity=0.5):
    """
    Generate a connected random graph with n nodes with black edges and controlled sparsity.

    Args:
        n: The number of nodes in the graph.
        sparsity: A value between 0 and 1 indicating the desired sparsity of the graph. 
                 0 represents a very sparse graph, while 1 represents a very dense graph.

    Returns:
        A NetworkX Graph object.
    """
    if n < 1:
        return nx.Graph()

    # Generate node names as uppercase letters A, B, C, ...
    node_names = [chr(65 + i) for i in range(n)]
    G = nx.Graph()
    G.add_nodes_from(node_names)

    # Create a spanning tree to ensure connectivity 
    for i in range(1, n):
        current_node = node_names[i]
        parent = node_names[random.randint(0, i-1)]
        weight = random.randint(1, 10)
        G.add_edge(current_node, parent, weight=weight, color='black')

    # Calculate the number of extra edges based on sparsity
    max_possible_edges = n * (n - 1) // 2  # Maximum possible edges in an undirected graph
    num_extra_edges = int(sparsity * (max_possible_edges - (n - 1))) 

    # Add additional random edges 
    for _ in range(num_extra_edges):
        u = random.choice(node_names)
        v = random.choice(node_names)
        if u != v and not G.has_edge(u, v):
            weight = random.randint(1, 10)
            G.add_edge(u, v, weight=weight, color='black')

    return G

def read_directed_graph(filename):
    """Reads a graph from a file and returns a NetworkX directed graph with black edges."""
    G = nx.DiGraph()
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            parts = line.split()
            if len(parts) != 3:
                print(f"Ignoring invalid line: {line}")
                continue
            node1, node2, weight_str = parts
            try:
                weight = int(weight_str)
            except ValueError:
                print(f"Invalid weight in line: {line}")
                continue
            # Add the edge with weight and default black color
            G.add_edge(node1, node2, weight=weight, color='black')
    return G

def read_undirected_graph(filename):
    """Reads a graph from a file and returns a NetworkX undirected graph."""
    G = nx.Graph()
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            parts = line.split()
            if len(parts) != 3:
                print(f"Ignoring invalid line: {line}")
                continue
            node1, node2, weight_str = parts
            try:
                weight = int(weight_str)
            except ValueError:
                print(f"Invalid weight in line: {line}")
                continue
            # Add the undirected edge to the graph
            G.add_edge(node1, node2, weight=weight, color='black')
    return G

def select_start_finish(G):
    """Randomly selects two distinct nodes from graph G as start and finish."""
    nodes = list(G.nodes())
    if len(nodes) < 2:
        raise ValueError("Graph must contain at least two nodes.")
    start, finish = random.sample(nodes, 2)
    return start, finish