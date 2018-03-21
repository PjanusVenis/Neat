from neuron_gene import NeuronGene
from connection_gene import ConnectionGene
from typing import Set, Dict


class Genome:
    def __init__(self, neuron_gene_list: Dict[int, NeuronGene], connection_gene_list: Set[ConnectionGene], birth_generation: int):
        self.neuron_gene_list = neuron_gene_list
        self.connection_gene_list = connection_gene_list
        self.birth_generation = birth_generation
