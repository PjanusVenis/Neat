from neuron_gene import NeuronGene
from connection_gene import ConnectionGene
from typing import Set, Dict


class Genome:
    def __init__(self, neuron_gene_list: Dict[int, NeuronGene], connection_gene_list: Set[ConnectionGene], birth_generation: int):
        self.neuron_gene_list = neuron_gene_list
        self.connection_gene_list = connection_gene_list
        self.birth_generation = birth_generation

    def is_connection_cyclic(self, from_id: int, to_id: int) -> bool:
        if from_id == to_id:
            return True

        src_neuron = self.neuron_gene_list[from_id]

        visited = [src_neuron]
        work_stack = [a for a in src_neuron.source_neurons]

        while len(work_stack) != 0:
            current_neuron: NeuronGene = work_stack.pop()

            if current_neuron in visited:
                continue

            if current_neuron.inno_id == to_id:
                return True

            visited.append(current_neuron)
            work_stack.extend([a for a in current_neuron.source_neurons])

        return False
