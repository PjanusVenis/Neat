from acyclic_network import AcyclicNetwork
from viznet import NodeBrush, EdgeBrush, DynamicShow

from connection_gene import ConnectionGene


def visualize_network(network: AcyclicNetwork):
    num_layers = len(network.layer_info)
    nodes_per_layer = [(network.layer_info[i + 1][0] - network.layer_info[i][0]) + 1 for i in range(num_layers - 1)]
    nodes_per_layer = [network.layer_info[0][0]] + nodes_per_layer
    max_num_nodes = max(nodes_per_layer)
    vertical_gap_size = 1.0 / (num_layers + 1)
    radius = 5

    with DynamicShow((10, 10), 'test.png') as d:
        brush = NodeBrush('nn.input', ax=d.ax)
        edge_brush = EdgeBrush('-', ax=d.ax, lw=2)

        nodes = []
        for y in range(num_layers):
            for x in range(nodes_per_layer[y]):
                horizontal_gap = 1.0 / (nodes_per_layer[y] + 1)
                nodes.append(brush >> ((x + 1) * horizontal_gap, vertical_gap_size * (y + 1)))

        conn_idx = 0

        for layer in network.layer_info:
            _, end_connection_idx = layer

            for idx in range(con_idx, end_connection_idx):
                from_id, to_id, _ = network.connections[idx]
                edge_brush >> (nodes[from_id], nodes[to_id])

            con_idx = end_connection_idx

