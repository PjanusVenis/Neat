from acyclic_network import AcyclicNetwork
from genome import Genome
from neuron_gene import NeuronGene


def create_acyclic_network(genome: Genome) -> AcyclicNetwork:
    return

def calculate_depth(genome: Genome) -> int:
    node_depth_by_id: {int: int} = {}
    for i in range(genome.)

def traverse_node(node: NeuronGene, node_depth_by_id: {int, int}, depth: int):
    if node.inno_id in node_depth_by_id:
        assigned_depth = node_depth_by_id[node.inno_id]
        if assigned_depth >= depth:
            return

    node_depth_by_id[node.inno_id] = depth

    for tgt_node in node.target_neurons:
        traverse_node(tgt_node, node_depth_by_id, depth + 1)