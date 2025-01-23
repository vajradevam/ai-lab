import heapq
import networkx as nx
import matplotlib.pyplot as plt

def uniform_cost_search(graph, start, goal):
    """
    Perform Uniform Cost Search on a graph from start to goal.
    
    Args:
        graph (nx.Graph): The graph to search.
        start (node): The starting node.
        goal (node): The goal node.
    
    Returns:
        tuple: (path, total_cost, graph) where path is a list of nodes,
               total_cost is the sum of the edge weights along the path,
               and graph is the input graph with path edges colored green.
    """
    if start not in graph or goal not in graph:
        return None, float('inf'), graph
    
    # Priority queue: (cumulative_cost, current_node, path)
    priority_queue = []
    heapq.heappush(priority_queue, (0, start, [start]))
    
    visited = {}  # Tracks the minimum cost to reach each node
    
    while priority_queue:
        current_cost, current_node, path = heapq.heappop(priority_queue)
        
        if current_node == goal:
            # Collect edges in both directions for undirected graphs
            path_edges = set()
            for i in range(len(path)-1):
                u, v = path[i], path[i+1]
                path_edges.add((u, v))
                path_edges.add((v, u))
            
            # Color all edges in the path
            for u, v in graph.edges():
                if (u, v) in path_edges:
                    graph[u][v]['color'] = 'green'
            
            return path, current_cost, graph
        
        if current_node in visited and visited[current_node] <= current_cost:
            continue
        visited[current_node] = current_cost
        
        for neighbor in graph.neighbors(current_node):
            edge_data = graph.get_edge_data(current_node, neighbor)
            edge_weight = edge_data.get('weight', 1)  # Default weight 1 if not present
            new_cost = current_cost + edge_weight
            new_path = path + [neighbor]
            
            if neighbor not in visited or new_cost < visited.get(neighbor, float('inf')):
                heapq.heappush(priority_queue, (new_cost, neighbor, new_path))
    
    # Goal not reachable
    return None, float('inf'), graph

