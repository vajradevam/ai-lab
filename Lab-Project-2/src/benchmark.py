import networkx as nx
import random
import time
import tracemalloc
import sys
import matplotlib.pyplot as plt
from solvers import *

def generate_test_graph(num_nodes=100, avg_degree=4):
    edges = set()
    while len(edges) < (avg_degree * num_nodes) // 2:
        u = random.randint(0, num_nodes - 1)
        v = random.randint(0, num_nodes - 1)
        if u != v:
            edges.add(tuple(sorted((u, v))))

    edge_list = []
    for u, v in edges:
        weight = random.randint(1, 10)  # Assign random weights
        edge_list.append((u, v, weight))

    graph = nx.Graph()
    for u, v, w in edge_list:
        graph.add_edge(u, v, weight=w)
    return edge_list, graph

def memory_usage():
    current, peak = tracemalloc.get_traced_memory()
    return current / 10**6, peak / 10**6  # Convert to MB

def run_and_measure(func, graph, start, goal, name):
    tracemalloc.start()
    start_time = time.time()
    path, cost, graph_result = func(graph.copy(), start, goal) #Important to pass a copy
    end_time = time.time()
    current_mem, peak_mem = memory_usage()
    tracemalloc.stop()
    print(f"{name} took {end_time - start_time:.6f} seconds, memory used: {current_mem:.2f} MB, peak memory: {peak_mem:.2f} MB, Path Cost: {cost}")
    return path, cost, graph_result, end_time - start_time, peak_mem

# Experiment parameters
num_nodes_list = [50, 100, 200, 400, 800, 1000, 1500]
avg_degree = 4

def simple():
    results = {}

    print("Nodes,Algorithm,Time (s),Peak Memory (MB),Path Length") #CSV Header

    for num_nodes in num_nodes_list:
        results[num_nodes] = {}
        edge_list, graph = generate_test_graph(num_nodes, avg_degree)
        start_node = 0
        goal_node = num_nodes - 1

        for search_name, search_func in {"BFS": breadth_first_search, "DFS": depth_first_search, "UCS": uniform_cost_search}.items():
            path, cost, graph_result, time_taken, peak_mem = run_and_measure(search_func, graph, start_node, goal_node, f"{search_name} (Nodes: {num_nodes})")
            results[num_nodes][search_name] = {"time": time_taken, "memory": peak_mem, "path_length": len(path) if path else 0}

            # Print results as comma-separated values (CSV)
            print(f"{num_nodes},{search_name},{time_taken:.6f},{peak_mem:.2f},{results[num_nodes][search_name]['path_length']}")

    # Plotting the results (optional, but still useful for visualization)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    for search_name in ["BFS", "DFS", "UCS"]:
        times = [results[n][search_name]["time"] for n in num_nodes_list]
        mem = [results[n][search_name]["memory"] for n in num_nodes_list]
        axes[0].plot(num_nodes_list, times, marker='o', label=search_name)
        axes[1].plot(num_nodes_list, mem, marker='o', label=search_name)

    axes[0].set_xlabel("Number of Nodes")
    axes[0].set_ylabel("Execution Time (seconds)")
    axes[0].set_title("Execution Time vs. Number of Nodes")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].set_xlabel("Number of Nodes")
    axes[1].set_ylabel("Peak Memory Usage (MB)")
    axes[1].set_title("Peak Memory Usage vs. Number of Nodes")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

    #Memory usage of graph objects
    edge_list, graph = generate_test_graph(100, 4)
    graph_from_list = nx.Graph()
    for u,v,w in edge_list:
        graph_from_list.add_edge(u,v, weight=w)
    print(f"\nSize of graph object in memory: {sys.getsizeof(graph)/10**6:.2f} MB")
    print(f"Size of graph_from_list object in memory: {sys.getsizeof(graph_from_list)/10**6:.2f} MB")
    print(f"Size of edge_list object in memory: {sys.getsizeof(edge_list)/10**6:.2f} MB")

simple()