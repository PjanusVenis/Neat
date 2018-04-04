from functools import cmp_to_key
from typing import List

from acyclic_network import AcyclicNetwork
from connection_gene import ConnectionGene
from genome import Genome
from neuron_gene import NeuronGene
from neuron_type import NeuronType


def create_acyclic_network(genome: Genome) -> AcyclicNetwork:

    max_depth, depth_info = calculate_depth(genome)
    node_count = len(genome.neuron_gene_list)
    input_bias_count = len(genome.input_gene_list)

    node_depths: List[(int, int, NeuronGene)] = [(depth_info[i], i, genome.neuron_gene_list[i]) for i in range(node_count)]
    node_depths_wo_inputs = node_depths[len(genome.input_gene_list):]
    node_depths_wo_inputs.sort(key=lambda x: x[0])
    node_depths[len(genome.input_gene_list):] = node_depths_wo_inputs

    idx_by_definition_idx: List[int] = [0 for _ in range(node_count)]
    idx_by_idx: {int: int} = {}

    for i in range(node_count):
        _, idx, neuron = node_depths[i]
        idx_by_definition_idx[idx] = i
        idx_by_idx[neuron.inno_id] = i

    output_count = len(genome.output_gene_list)
    output_neuron_idx_list = idx_by_definition_idx[input_bias_count:input_bias_count + output_count]

    activation_fns = [genome.neuron_gene_list[node_depths[i][1]].activation_fn for a in range(node_count)]

    connection_list: List[(int, int, float)] = []

    for i in range(len(genome.connection_gene_list)):
        connection = genome.connection_gene_list[i]
        connection_list.append((idx_by_idx[connection.from_id], idx_by_idx[connection.to_id], connection.weight))

    def connection_comparator(a, b):
        from_a, to_a, _ = a
        from_b, to_b, _ = b

        if from_a < from_b:
            return -1
        if from_a > from_b:
            return 1
        if to_a < to_b:
            return -1
        if to_a > to_b:
            return 1

        return 0

    connection_list.sort(key=cmp_to_key(connection_comparator))

    node_idx = input_bias_count
    conn_idx = 0

    layers_info = []

    for depth in range(max_depth + 1):
        while node_idx < node_count and node_depths[node_idx][0] == depth:
            node_idx += 1
        while conn_idx < len(connection_list) and node_depths[connection_list[conn_idx][0]][0] == depth:
            conn_idx += 1

        layers_info.append((node_idx, conn_idx))

    return AcyclicNetwork(activation_fns, connection_list, layers_info, output_neuron_idx_list, node_count, input_bias_count - 1, output_count)


def calculate_depth(genome: Genome) -> (int, List[int]):
    node_depth_by_id: {int: int} = {}
    input_bias_count = len(genome.input_gene_list)

    for i in range(input_bias_count):
        traverse_node(genome.neuron_gene_list[i], node_depth_by_id, 0)

    for i in range(input_bias_count, len(genome.neuron_gene_list)):
        if genome.neuron_gene_list[i].inno_id not in node_depth_by_id:
            node_depth_by_id[genome.neuron_gene_list[i].inno_id] = 0

    node_depth_info = [node_depth_by_id[a.inno_id] for a in genome.neuron_gene_list]
    max_depth = max(node_depth_info)

    return max_depth, node_depth_info


def traverse_node(node: NeuronGene, node_depth_by_id: {int, int}, depth: int):
    if node.inno_id in node_depth_by_id:
        assigned_depth = node_depth_by_id[node.inno_id]
        if assigned_depth >= depth:
            return

    node_depth_by_id[node.inno_id] = depth

    for tgt_node in node.target_neurons:
        traverse_node(tgt_node, node_depth_by_id, depth + 1)