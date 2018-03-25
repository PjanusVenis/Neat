from typing import List

from genome import Genome


class Specie:
    def __init__(self):
        self.genome_list: List[Genome] = []
        self.centroid = []

    def total_fitness(self) -> float:
        return sum([a.evaluation.fitness for a in self.genome_list])

    def total_novelty(self) -> float:
        return sum([a.evaluation.novelty for a in self.genome_list])

    def mean_fitness(self) -> float:
        return self.total_fitness() / len(self.genome_list)

    def mean_novelty(self) -> float:
        return self.total_novelty() / len(self.genome_list)

    def total_complexity(self) -> float:
        return sum([len(a.neuron_gene_list) + len(a.connection_gene_list) for a in self.genome_list])

    def mean_complexity(self) -> float:
        return self.total_complexity() / len(self.genome_list)