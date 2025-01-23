import matplotlib.pyplot as plt
import networkx as nx

def visualize_ucs_path(G, path, cost, pos=None, seed=22051662, k=0.5):
    """
    Visualizes a graph with the Uniform Cost Search (UCS) path highlighted.
    
    Parameters:
    G (NetworkX Graph): The graph to visualize
    path (list): List of nodes representing the optimal path
    cost (numeric): Total cost of the optimal path
    pos (dict, optional): Node positions. If None, will be generated using spring layout
    seed (int): Random seed for reproducible layout
    k (float): Optimal distance between nodes for spring layout
    """
    plt.figure(figsize=(14, 12))
    
    # Generate positions if not provided
    if pos is None:
        pos = nx.spring_layout(G, seed=seed, k=k)

    # Style setup
    start_node, end_node = path[0], path[-1]
    node_colors = [
        'limegreen' if n == start_node else
        'tomato' if n == end_node else
        '#80b3ff' for n in G.nodes()
    ]
    path_edges = list(zip(path[:-1], path[1:]))

    # Edge styling
    edge_colors = []
    edge_widths = []
    for u, v in G.edges():
        if (u, v) in path_edges or (v, u) in path_edges:
            edge_colors.append('red')
            edge_widths.append(4)
        else:
            edge_colors.append('gray')
            edge_widths.append(1.2)

    # Draw main elements
    nx.draw_networkx_nodes(
        G, pos,
        node_size=1500,
        node_color=node_colors,
        edgecolors='black',
        linewidths=3,
        alpha=0.95
    )

    # Add glow effect for start/end
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=[start_node, end_node],
        node_size=1700,
        node_color=[node_colors[0], node_colors[-1]],
        edgecolors='gold',
        linewidths=4,
        alpha=0.6
    )

    nx.draw_networkx_edges(
        G, pos,
        edge_color=edge_colors,
        width=edge_widths,
        alpha=0.8
    )

    # Labels
    nx.draw_networkx_labels(
        G, pos,
        font_size=16,
        font_weight='bold',
        font_family='sans-serif',
        font_color='black'
    )

    edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=11,
        font_weight='bold',
        font_color='darkgreen',
        bbox=dict(alpha=0)
    )

    # Enhanced legend
    legend_elements = [
        plt.Line2D([0], [0], color='red', lw=4, label='UCS Path'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='limegreen',
                   markersize=20, markeredgecolor='black', label=f'Start: {start_node}'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='tomato',
                   markersize=20, markeredgecolor='black', label=f'End: {end_node}'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#80b3ff',
                   markersize=15, label='Other Nodes'),
        plt.Line2D([0], [0], color='gray', lw=2, label='Other Paths')
    ]

    plt.legend(
        handles=legend_elements,
        loc='upper right',
        title="Graph Components:",
        title_fontsize=13,
        frameon=True,
        framealpha=0.95,
        edgecolor='black'
    )

    # Final styling
    plt.title(f"Uniform Cost Search: {start_node} → {end_node} (Cost: {cost})", 
             fontsize=18, pad=20, fontweight='bold')
    plt.gca().set_facecolor('#f0f0f0')
    plt.grid(False)
    plt.axis('off')

    # Output path information
    print(f"Optimal Path: {' → '.join(path)}")
    print(f"Total Cost: {cost}")
    
    plt.tight_layout()
    plt.show()