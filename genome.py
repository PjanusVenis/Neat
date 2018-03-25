from evalution_info import EvaluationInfo
from neuron_gene import NeuronGene
from connection_gene import ConnectionGene
from typing import Set, Dict, List

from neuron_type import NeuronType


class Genome:
    def __init__(self, neuron_gene_dict: Dict[int, NeuronGene], neuron_gene_list: List[NeuronGene], connection_gene_list: Set[ConnectionGene], birth_generation: int):
        self.neuron_gene_dict = neuron_gene_dict
        self.connection_gene_list = connection_gene_list
        self.birth_generation = birth_generation
        self.neuron_gene_list = neuron_gene_list
        self.input_gene_list = {n for _, n in neuron_gene_dict.items() if n.type == NeuronType.BIAS or n.type == NeuronType.INPUT}
        self.output_gene_list = {n for _, n in neuron_gene_dict.items() if n.type == NeuronType.OUTPUT}
        self.hidden_gene_list = {n for _, n in neuron_gene_dict.items() if n.type == NeuronType.HIDDEN}
        self.evaluation: EvaluationInfo = EvaluationInfo(0, 0)
        self.position: List[(int, float)] = []

    def is_connection_cyclic(self, from_id: int, to_id: int) -> bool:
        if from_id == to_id:
            return True

        src_neuron = self.neuron_gene_dict[from_id]

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

    def get_position(self):
        if len(self.position) == 0:
            self.position = [(a.inno_id, a.weight) for a in self.connection_gene_list]
        return self.position